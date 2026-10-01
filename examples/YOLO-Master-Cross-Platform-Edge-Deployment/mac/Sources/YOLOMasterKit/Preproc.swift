// GPU preprocessing on Apple platforms: one Metal compute kernel does letterbox (bilinear, the
// cv::resize INTER_LINEAR source-coordinate rule the Linux runtime and its CUDA kernel use),
// BGRA/RGBA -> planar RGB / 255, 114-gray padding, straight into a shared-storage MTLBuffer that
// Core ML reads as the input MLMultiArray without a CPU copy. Replaces the CGContext draw + vDSP
// planar conversion of the CPU path (Detector.letterbox / fillInput).
//
// Differences from the CPU path, on purpose: integer pads ((imgsz - out) / 2, like the C++
// letterbox_params) instead of the CGContext's fractional centring, and bilinear sampling instead
// of Core Graphics' interpolation, so the tensor can be checked against the Linux preprocess_nchw
// within 1/255 (scripts/preproc_compare.py on a --dump-input blob). Core ML still copies the
// multiarray into its own storage; what this removes is the CPU resample and the u8 -> f32 loop.
//
// The kernel is compiled at runtime from the string below (`swift build` does not compile .metal
// files; Xcode would, but one path that works in every build is better), once per device.
import Foundation
import CoreGraphics
import CoreML
import CoreVideo
import Metal

public enum PreprocDevice: String, CaseIterable, Sendable { case cpu, gpu }

/// Geometry of a preprocessed frame in the integer-pad convention.
public struct PreprocGeometry: Sendable {
    public let srcW, srcH, outW, outH, padX, padY, imgsz: Int
    public let scaleX, scaleY: CGFloat
}

public final class MetalPreprocessor {
    private struct Params {   // mirrors the MSL struct below (all 4-byte members, no padding surprises)
        var src_w: Int32, src_h: Int32, out_w: Int32, out_h: Int32, pad_x: Int32, pad_y: Int32, imgsz: Int32
        var fx: Float, fy: Float
    }
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct PreprocParams { int src_w, src_h, out_w, out_h, pad_x, pad_y, imgsz; float fx, fy; };
    // one thread per destination pixel, three planes written. Inside the resized rectangle the
    // source coordinate is (dst + 0.5) * f - 0.5 clamped at 0 (cv::resize INTER_LINEAR); outside it
    // is the 114 letterbox gray. Unorm texture reads already give value / 255 in RGBA order
    // whatever the storage order (bgra8 / rgba8), so no swizzle is needed.
    kernel void letterbox_nchw(texture2d<float, access::read> src [[texture(0)]],
                               device float* dst [[buffer(0)]],
                               constant PreprocParams& p [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= uint(p.imgsz) || gid.y >= uint(p.imgsz)) return;
        const int hw = p.imgsz * p.imgsz;
        const int idx = int(gid.y) * p.imgsz + int(gid.x);
        float3 c;
        const int rx = int(gid.x) - p.pad_x, ry = int(gid.y) - p.pad_y;
        if (rx < 0 || ry < 0 || rx >= p.out_w || ry >= p.out_h) {
            c = float3(114.0f / 255.0f);
        } else {
            float sx = (rx + 0.5f) * p.fx - 0.5f;
            float sy = (ry + 0.5f) * p.fy - 0.5f;
            sx = max(sx, 0.0f); sy = max(sy, 0.0f);
            int x0 = min(int(sx), p.src_w - 1), y0 = min(int(sy), p.src_h - 1);
            const int x1 = min(x0 + 1, p.src_w - 1), y1 = min(y0 + 1, p.src_h - 1);
            const float wx = sx - float(x0), wy = sy - float(y0);
            const float3 c00 = src.read(uint2(x0, y0)).rgb, c01 = src.read(uint2(x1, y0)).rgb;
            const float3 c10 = src.read(uint2(x0, y1)).rgb, c11 = src.read(uint2(x1, y1)).rgb;
            c = c00 * ((1.0f - wx) * (1.0f - wy)) + c01 * (wx * (1.0f - wy)) + c10 * ((1.0f - wx) * wy) + c11 * (wx * wy);
        }
        dst[idx] = c.r; dst[hw + idx] = c.g; dst[2 * hw + idx] = c.b;
    }
    """

    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var textureCache: CVMetalTextureCache?
    private var ring: [MTLBuffer] = []     // two output buffers: one may still be read by Core ML while the next is written
    private var ringIndex = 0
    private var uploadTexture: MTLTexture?  // reused for same-size CGImage uploads

    static var available: Bool { MTLCreateSystemDefaultDevice() != nil }

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = false          // keep the bilinear arithmetic IEEE so the parity check is meaningful
        guard let lib = try? dev.makeLibrary(source: MetalPreprocessor.source, options: opts),
              let fn = lib.makeFunction(name: "letterbox_nchw"),
              let ps = try? dev.makeComputePipelineState(function: fn) else { return nil }
        device = dev; queue = q; pipeline = ps
        CVMetalTextureCacheCreate(nil, nil, dev, nil, &textureCache)
    }

    static func geometry(srcW: Int, srcH: Int, imgsz: Int, stretch: Bool) -> PreprocGeometry {
        if stretch {
            return PreprocGeometry(srcW: srcW, srcH: srcH, outW: imgsz, outH: imgsz, padX: 0, padY: 0, imgsz: imgsz,
                                   scaleX: CGFloat(imgsz) / CGFloat(srcW), scaleY: CGFloat(imgsz) / CGFloat(srcH))
        }
        let r = min(CGFloat(imgsz) / CGFloat(srcW), CGFloat(imgsz) / CGFloat(srcH))
        let ow = Int((CGFloat(srcW) * r).rounded()), oh = Int((CGFloat(srcH) * r).rounded())
        return PreprocGeometry(srcW: srcW, srcH: srcH, outW: ow, outH: oh, padX: (imgsz - ow) / 2, padY: (imgsz - oh) / 2,
                               imgsz: imgsz, scaleX: r, scaleY: r)
    }

    // ---- sources ----
    /// The pixels a CGImage contributes to the kernel: 32-bit RGBA / BGRA providers as they are, anything
    /// else rendered once into an RGBA raster at native size (a copy, not a resample). Public so the
    /// parity dump can write exactly what the kernel saw (`--dump-input`).
    public struct SourcePixels {
        public let data: Data
        public let width, height, bytesPerRow: Int
        public let bgra: Bool          // byte order in `data`: BGRA when true, RGBA otherwise
    }
    public static func sourcePixels(_ image: CGImage) -> SourcePixels? {
        let w = image.width, h = image.height
        if image.bitsPerPixel == 32, image.bitsPerComponent == 8, let data = image.dataProvider?.data as Data? {
            let alpha = image.alphaInfo
            let little = image.bitmapInfo.intersection(.byteOrderMask) == .byteOrder32Little
            let first = alpha == .premultipliedFirst || alpha == .noneSkipFirst || alpha == .first
            let last = alpha == .premultipliedLast || alpha == .noneSkipLast || alpha == .last
            if little && first { return SourcePixels(data: data, width: w, height: h, bytesPerRow: image.bytesPerRow, bgra: true) }
            if !little && last { return SourcePixels(data: data, width: w, height: h, bytesPerRow: image.bytesPerRow, bgra: false) }
        }
        var px = [UInt8](repeating: 0, count: w * h * 4)
        let ok = px.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? SourcePixels(data: Data(px), width: w, height: h, bytesPerRow: w * 4, bgra: false) : nil
    }

    func texture(from image: CGImage) -> MTLTexture? {
        guard let src = MetalPreprocessor.sourcePixels(image) else { return nil }
        let w = src.width, h = src.height
        let format: MTLPixelFormat = src.bgra ? .bgra8Unorm : .rgba8Unorm
        let bytesPerRow = src.bytesPerRow
        let data = src.data
        if uploadTexture == nil || uploadTexture!.width != w || uploadTexture!.height != h || uploadTexture!.pixelFormat != format {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            uploadTexture = device.makeTexture(descriptor: d)
        }
        guard let tex = uploadTexture else { return nil }
        data.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bytesPerRow)
        }
        return tex
    }

    /// Zero-copy texture over a BGRA camera / reader pixel buffer (IOSurface-backed, Metal compatible).
    func texture(from pixelBuffer: CVPixelBuffer) -> (MTLTexture, CVMetalTexture)? {
        guard let cache = textureCache, CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else { return nil }
        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        var cvtex: CVMetalTexture?
        let st = CVMetalTextureCacheCreateTextureFromImage(nil, cache, pixelBuffer, nil, .bgra8Unorm, w, h, 0, &cvtex)
        guard st == kCVReturnSuccess, let ct = cvtex, let tex = CVMetalTextureGetTexture(ct) else { return nil }
        return (tex, ct)
    }

    // ---- the pass ----
    struct Output {
        let array: MLMultiArray
        let geometry: PreprocGeometry
        let gpuMs: Double        // command buffer GPU time
    }

    private func outputBuffer(_ imgsz: Int) -> MTLBuffer? {
        let length = 3 * imgsz * imgsz * MemoryLayout<Float>.size
        if ring.count < 2 || ring[0].length != length {
            ring = (0..<2).compactMap { _ in device.makeBuffer(length: length, options: .storageModeShared) }
            if ring.count < 2 { return nil }
        }
        ringIndex = (ringIndex + 1) % 2
        return ring[ringIndex]
    }

    func run(texture: MTLTexture, srcW: Int, srcH: Int, imgsz: Int, stretch: Bool) -> Output? {
        let g = MetalPreprocessor.geometry(srcW: srcW, srcH: srcH, imgsz: imgsz, stretch: stretch)
        guard let out = outputBuffer(imgsz), let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        var p = Params(src_w: Int32(srcW), src_h: Int32(srcH), out_w: Int32(g.outW), out_h: Int32(g.outH),
                       pad_x: Int32(g.padX), pad_y: Int32(g.padY), imgsz: Int32(imgsz),
                       fx: Float(srcW) / Float(g.outW), fy: Float(srcH) / Float(g.outH))
        enc.setComputePipelineState(pipeline)
        enc.setTexture(texture, index: 0)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        let tw = pipeline.threadExecutionWidth
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        let tg = MTLSize(width: tw, height: th, depth: 1)
        let groups = MTLSize(width: (imgsz + tw - 1) / tw, height: (imgsz + th - 1) / th, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()          // Core ML has no cross-queue sync API: the tensor must be complete here
        guard cb.status == .completed else { return nil }
        let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        let hw = imgsz * imgsz
        let keep = out
        guard let arr = try? MLMultiArray(dataPointer: out.contents(), shape: [1, 3, NSNumber(value: imgsz), NSNumber(value: imgsz)],
                                          dataType: .float32,
                                          strides: [NSNumber(value: 3 * hw), NSNumber(value: hw), NSNumber(value: imgsz), 1],
                                          deallocator: { _ in _ = keep }) else { return nil }
        return Output(array: arr, geometry: g, gpuMs: gpuMs)
    }
}

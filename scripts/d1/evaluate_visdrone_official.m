function report = evaluate_visdrone_official(toolkit, dataset, predictions, output, toolkitManifest)
% Invoke unmodified pinned matching/AP routines; only normalize empty TXT input.
expectedCommit = '005445782213e20cb91bc50a597db3dd949e749a';
manifest = jsondecode(fileread(toolkitManifest));
assert(strcmp(manifest.commit, expectedCommit), 'Unexpected toolkit commit');
for k = 1:numel(manifest.files)
    item = manifest.files(k);
    assert(strcmp(sha256file(fullfile(toolkit, item.path)), item.sha256), 'Toolkit file checksum mismatch');
end
oldPath = path;
cleanup = onCleanup(@() path(oldPath));
addpath(fullfile(toolkit, 'utils'), '-begin');
if exist('mean2', 'file') == 0
    addpath(fullfile(fileparts(mfilename('fullpath')), 'visdrone_matlab_compat'), '-begin');
end
gtPath = fullfile(dataset, 'annotations');
imgPath = fullfile(dataset, 'images');
files = dir(fullfile(gtPath, '*.txt'));
names = sort({files.name});
assert(~isempty(names), 'Empty evaluation set');
detFiles = dir(fullfile(predictions, '*.txt'));
assert(isequal(sort({detFiles.name}), names), 'Prediction file coverage differs from GT');
allgt = cell(1, numel(names));
alldet = cell(1, numel(names));
for k = 1:numel(names)
    gtFile = fullfile(gtPath, names{k});
    detFile = fullfile(predictions, names{k});
    gt = readBoxes(gtFile);
    dt = readBoxes(detFile);
    assert(all(ismember(dt(:,6), 1:10)), 'Prediction class must be 1..10');
    assert(size(dt,1) <= 500 && all(diff(dt(:,5)) <= 0), 'Detections must be sorted top500');
    img = imread(fullfile(imgPath, [names{k}(1:end-4), '.jpg']));
    [newgt, dt] = dropObjectsInIgr(gt, dt, size(img,1), size(img,2));
    gt = newgt;
    gt(newgt(:,5) == 0, 5) = 1;
    gt(newgt(:,5) == 1, 5) = 0;
    allgt{k} = gt;
    alldet{k} = dt;
end
[a,b,c,d,e,f,g] = calcAccuracy(numel(names), allgt, alldet);
metricNames = {'AP_all','AP_50','AP_75','AR_1','AR_10','AR_100','AR_500'};
values = [a,b,c,d,e,f,g];
assert(all(isfinite(values)) && all(values >= 0 & values <= 100), 'Invalid official result');
report.backend = 'official-matlab';
report.toolkit_commit = expectedCommit;
report.matlab_version = version;
report.image_count = numel(names);
report.metrics_percent = cell2struct(num2cell(values), metricNames, 2);
report.metrics = cell2struct(num2cell(values / 100), metricNames, 2);
report.toolkit_manifest_sha256 = sha256file(toolkitManifest);
report.export_sha256 = sha256file(fullfile(predictions, 'export.json'));
report.empty_input_policy = 'empty TXT becomes zeros(0,8); matching/AP routines unchanged';
report.mean2_implementation = which('mean2');
fid = fopen(output, 'w');
assert(fid >= 0, 'Cannot write result');
closeFile = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, PrettyPrint=true));
end

function boxes = readBoxes(filename)
text = strtrim(fileread(filename));
if isempty(text)
    boxes = zeros(0,8);
else
    boxes = readmatrix(filename, FileType='text', Delimiter=',');
    if size(boxes,2) == 9 && all(isnan(boxes(:,9)))
        boxes = boxes(:,1:8);
    end
    assert(size(boxes,2) == 8 && all(isfinite(boxes), 'all'), 'Invalid annotation/detection');
end
end

function value = sha256file(filename)
fid = fopen(filename, 'rb');
assert(fid >= 0, 'Missing toolkit file');
guard = onCleanup(@() fclose(fid));
md = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    md.update(fread(fid, 1048576, '*uint8'));
end
value = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'),2).',1,[]));
end

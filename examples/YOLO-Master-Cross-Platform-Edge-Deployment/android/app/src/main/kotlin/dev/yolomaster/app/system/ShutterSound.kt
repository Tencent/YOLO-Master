package dev.yolomaster.app.system

import android.media.MediaActionSound

/** The system shutter click (iOS `AudioServicesPlaySystemSound(1108)`). */
class ShutterSound {
    private val sound = MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) }
    fun play() = sound.play(MediaActionSound.SHUTTER_CLICK)
    fun release() = sound.release()
}

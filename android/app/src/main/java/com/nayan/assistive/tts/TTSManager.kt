package com.nayan.assistive.tts

import android.content.Context
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import com.nayan.assistive.priority.HazardPriorityEngine.HazardScore
import java.util.Locale

/**
 * Text-to-Speech Alert Manager for NAYAN.
 *
 * Implements:
 * 1. Audio alert cooldown to prevent cognitive overload.
 * 2. Unclear camera position alert when camera is blocked/obscured.
 * 3. Contextual hazard alerts: e.g., "Car approaching fast", "Stairs ahead in path".
 */
class TTSManager(context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isInitialized = false

    private var lastAlertTimestamp = 0L
    private val alertCooldownMillis = 2500L // 2.5s cooldown between spoken alerts

    private var emptyStreak = 0
    private val emptyStreakThreshold = 20 // ~0.7s of missing frames triggers obstruction check

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTSManager", "Language not supported by TextToSpeech.")
            } else {
                tts?.setSpeechRate(1.1f)
                tts?.setPitch(1.0f)
                isInitialized = true
                speak("NAYAN assistive navigation active.")
            }
        } else {
            Log.e("TTSManager", "TextToSpeech initialization failed.")
        }
    }

    fun speak(text: String, force: Boolean = false) {
        if (!isInitialized) return
        val now = SystemClock.elapsedRealtime()

        if (force || (now - lastAlertTimestamp >= alertCooldownMillis)) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "NAYAN_ALERT")
            lastAlertTimestamp = now
        }
    }

    /**
     * Called on every frame with the evaluated top hazard (or null if none).
     */
    fun processHazardAlert(topHazard: HazardScore?) {
        val now = SystemClock.elapsedRealtime()

        if (topHazard == null) {
            emptyStreak++
            if (emptyStreak >= emptyStreakThreshold && (now - lastAlertTimestamp >= alertCooldownMillis)) {
                // Periodically notify user if camera might be blocked
                if (emptyStreak % (emptyStreakThreshold * 3) == 0) {
                    speak("Camera view clear or path unobstructed.")
                }
            }
            return
        }

        emptyStreak = 0

        // Only announce if cooldown has expired
        if (now - lastAlertTimestamp < alertCooldownMillis) return

        val className = topHazard.detection.className
        val isCriticalTtc = (topHazard.ttcSeconds != null && topHazard.ttcSeconds <= 1.5f)
        val isInCorridor = topHazard.pathScore > 0.4f

        val message = when {
            isCriticalTtc -> "Warning! $className approaching fast!"
            isInCorridor && (className.equals("stairs", ignoreCase = true) || className.equals("pothole", ignoreCase = true)) ->
                "Caution! $className directly in walking path."
            isInCorridor -> "$className ahead in path."
            else -> "$className nearby on side."
        }

        speak(message)
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}

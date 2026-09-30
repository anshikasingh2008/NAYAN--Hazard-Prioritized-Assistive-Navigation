package com.nayan.assistive.priority

import android.graphics.RectF
import com.nayan.assistive.ml.Detection
import com.nayan.assistive.tracking.ObjectTracker
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Port of NAYAN Hazard Prioritization Engine to Kotlin.
 *
 * Combines 5 key metrics:
 * 1. Proximity Score (bounding box size relative to camera view)
 * 2. Motion Score (displacement over consecutive frames)
 * 3. Path-Awareness / Walking Corridor (central 40% walking path)
 * 4. Time-to-Collision (TTC estimated via bounding box area expansion)
 * 5. Class Risk Weights (calibrated for pedestrian navigation, including stairs and potholes)
 */
class HazardPriorityEngine {

    companion object {
        // Base risk weights across all 13 supported classes
        val RISK_WEIGHTS: Map<String, Float> = mapOf(
            "person" to 0.5f,        // Index 0
            "bicycle" to 0.8f,       // Index 1
            "car" to 1.0f,           // Index 2
            "motorcycle" to 1.0f,    // Index 3
            "bus" to 1.0f,           // Index 4
            "truck" to 1.0f,         // Index 5
            "traffic light" to 0.2f, // Index 6
            "fire hydrant" to 0.3f,  // Index 7
            "stop sign" to 0.3f,     // Index 8
            "bench" to 0.2f,         // Index 9
            "chair" to 0.2f,         // Index 10
            "stairs" to 1.0f,        // Index 11 (Critical fall hazard)
            "pothole" to 0.9f        // Index 12 (Critical ground hazard)
        )

        const val DEFAULT_RISK_WEIGHT: Float = 0.3f
        const val DANGER_AREA_RATIO: Float = 0.15f // 15% of frame area is immediate collision zone
    }

    data class HazardScore(
        val detection: Detection,
        val finalScore: Float,
        val proximityScore: Float,
        val motionScore: Float,
        val pathScore: Float,
        val ttcScore: Float,
        val ttcSeconds: Float?,
        val riskWeight: Float
    )

    // -------------------------------------------------------------
    // 1. PROXIMITY SCORE
    // -------------------------------------------------------------
    fun calculateProximityScore(bbox: RectF, frameWidth: Float, frameHeight: Float): Float {
        val boxArea = bbox.width() * bbox.height()
        val frameArea = frameWidth * frameHeight
        if (frameArea <= 0f) return 0f
        return min((boxArea / frameArea) * 4.0f, 1.0f)
    }

    // -------------------------------------------------------------
    // 2. MOTION SCORE
    // -------------------------------------------------------------
    fun calculateMotionScore(
        currentCenter: Pair<Float, Float>,
        previousCenter: Pair<Float, Float>?,
        frameWidth: Float
    ): Float {
        if (previousCenter == null || frameWidth <= 0f) return 0f
        val dx = currentCenter.first - previousCenter.first
        val dy = currentCenter.second - previousCenter.second
        val distance = sqrt(dx * dx + dy * dy)
        return min(distance / (frameWidth * 0.1f), 1.0f)
    }

    // -------------------------------------------------------------
    // 3. PATH RELEVANCE SCORE (Central 40% Walking Corridor)
    // -------------------------------------------------------------
    fun calculatePathRelevanceScore(bbox: RectF, frameWidth: Float): Float {
        val centerX = bbox.centerX()
        val corridorLeft = frameWidth * 0.30f
        val corridorRight = frameWidth * 0.70f

        // Outside corridor gets 0.0
        if (centerX < corridorLeft || centerX > corridorRight) {
            return 0.0f
        }

        val corridorCenter = frameWidth * 0.50f
        val corridorHalfWidth = frameWidth * 0.20f
        val distance = abs(centerX - corridorCenter)

        val score = 1.0f - (distance / corridorHalfWidth)
        return max(0.0f, min(score, 1.0f))
    }

    // -------------------------------------------------------------
    // 4. TIME-TO-COLLISION (TTC) SCORE
    // -------------------------------------------------------------
    fun calculateTtcScore(
        currentBbox: RectF,
        previousBbox: RectF?,
        frameWidth: Float,
        frameHeight: Float,
        fps: Float = 30f
    ): Pair<Float, Float?> {
        if (previousBbox == null) {
            return Pair(0.0f, null)
        }

        val currentArea = currentBbox.width() * currentBbox.height()
        val previousArea = previousBbox.width() * previousBbox.height()
        val areaGrowth = currentArea - previousArea

        // Object is not expanding (moving away or stationary)
        if (areaGrowth <= 0f) {
            return Pair(0.0f, null)
        }

        val frameArea = frameWidth * frameHeight
        val dangerArea = frameArea * DANGER_AREA_RATIO

        // Already reached collision threshold
        if (currentArea >= dangerArea) {
            return Pair(1.0f, 0.0f)
        }

        val remainingArea = dangerArea - currentArea
        val ttcFrames = remainingArea / areaGrowth
        val ttcSeconds = ttcFrames / fps

        val urgencyScore = when {
            ttcSeconds <= 0.5f -> 1.0f
            ttcSeconds <= 1.0f -> 0.9f
            ttcSeconds <= 2.0f -> 0.7f
            ttcSeconds <= 3.0f -> 0.4f
            ttcSeconds <= 5.0f -> 0.2f
            else -> 0.0f
        }

        return Pair(urgencyScore, ttcSeconds)
    }

    // -------------------------------------------------------------
    // 5. SCORE INDIVIDUAL DETECTION
    // -------------------------------------------------------------
    fun scoreDetection(
        detection: Detection,
        frameWidth: Float,
        frameHeight: Float,
        previousCenter: Pair<Float, Float>?,
        previousBbox: RectF?,
        fps: Float = 30f
    ): HazardScore {
        val currentCenter = Pair(detection.bbox.centerX(), detection.bbox.centerY())

        val pScore = calculateProximityScore(detection.bbox, frameWidth, frameHeight)
        val mScore = calculateMotionScore(currentCenter, previousCenter, frameWidth)
        val pathScore = calculatePathRelevanceScore(detection.bbox, frameWidth)
        val (ttcScore, ttcSeconds) = calculateTtcScore(detection.bbox, previousBbox, frameWidth, frameHeight, fps)

        val riskWeight = RISK_WEIGHTS[detection.className.lowercase()] ?: DEFAULT_RISK_WEIGHT

        // Base linear combination
        val baseScore = (0.30f * pScore) +
                (0.15f * mScore) +
                (0.20f * riskWeight) +
                (0.15f * pathScore) +
                (0.20f * ttcScore)

        // Path multiplier: objects directly ahead get up to 50% boost
        val pathMultiplier = 1.0f + (0.50f * pathScore)
        val finalScore = min(baseScore * pathMultiplier, 1.0f)

        return HazardScore(
            detection = detection,
            finalScore = finalScore,
            proximityScore = pScore,
            motionScore = mScore,
            pathScore = pathScore,
            ttcScore = ttcScore,
            ttcSeconds = ttcSeconds,
            riskWeight = riskWeight
        )
    }

    // -------------------------------------------------------------
    // 6. IDENTIFY TOP HAZARD
    // -------------------------------------------------------------
    fun getTopHazard(
        detections: List<Detection>,
        tracker: ObjectTracker,
        frameWidth: Float,
        frameHeight: Float,
        confidenceThreshold: Float = 0.35f,
        fps: Float = 30f
    ): HazardScore? {
        val validDetections = detections.filter { it.confidence >= confidenceThreshold }
        if (validDetections.isEmpty()) return null

        val scoredList = validDetections.map { det ->
            val track = det.trackId?.let { tracker.getTrack(it) }
            val prevCenter = track?.previousCenter
            val prevBbox = track?.previousBbox

            scoreDetection(det, frameWidth, frameHeight, prevCenter, prevBbox, fps)
        }

        return scoredList.maxByOrNull { it.finalScore }
    }
}

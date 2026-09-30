package com.nayan.assistive.tracking

import android.graphics.RectF
import com.nayan.assistive.ml.Detection
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Lightweight Real-Time Multi-Object Tracker for Mobile.
 *
 * Solves the problem of associating multiple objects of the same class
 * across frames by tracking spatial proximity and IoU (Intersection-over-Union).
 *
 * Each tracked object maintains:
 * - A persistent unique `trackId`
 * - History of bounding boxes and centers across frames
 * - Disappearance counter (grace period before track is dropped)
 */
class ObjectTracker(
    private val iouThreshold: Float = 0.25f,
    private val maxMissingFrames: Int = 10
) {
    private var nextTrackId = 1
    private val activeTracks = mutableMapOf<Int, Track>()

    data class Track(
        val id: Int,
        var bbox: RectF,
        var classId: Int,
        var className: String,
        var missingFrames: Int = 0,
        var previousBbox: RectF? = null,
        var previousCenter: Pair<Float, Float>? = null
    ) {
        fun currentCenter(): Pair<Float, Float> {
            return Pair(bbox.centerX(), bbox.centerY())
        }
    }

    /**
     * Updates active tracks with new detections from the current frame.
     * Assigns persistent `trackId`s to detections.
     */
    fun update(detections: List<Detection>): List<Detection> {
        val matchedDetectionIndices = mutableSetOf<Int>()
        val matchedTrackIds = mutableSetOf<Int>()

        // 1. Try to match existing tracks with current detections using IoU and centroid distance
        for ((trackId, track) in activeTracks) {
            var bestIou = 0f
            var bestMatchIdx = -1

            for (i in detections.indices) {
                if (matchedDetectionIndices.contains(i)) continue
                val det = detections[i]

                // Class must match to associate
                if (det.classId != track.classId) continue

                val iou = calculateIoU(track.bbox, det.bbox)
                if (iou > bestIou && iou >= iouThreshold) {
                    bestIou = iou
                    bestMatchIdx = i
                }
            }

            // Fallback: If IoU is low (e.g. fast motion), check centroid distance
            if (bestMatchIdx == -1) {
                var minDistance = Float.MAX_VALUE
                val trackCenter = track.currentCenter()

                for (i in detections.indices) {
                    if (matchedDetectionIndices.contains(i)) continue
                    val det = detections[i]
                    if (det.classId != track.classId) continue

                    val dx = det.bbox.centerX() - trackCenter.first
                    val dy = det.bbox.centerY() - trackCenter.second
                    val dist = sqrt(dx * dx + dy * dy)

                    // Must be within reasonable movement radius (e.g. 150px)
                    if (dist < 150f && dist < minDistance) {
                        minDistance = dist
                        bestMatchIdx = i
                    }
                }
            }

            if (bestMatchIdx != -1) {
                val det = detections[bestMatchIdx]
                track.previousBbox = RectF(track.bbox)
                track.previousCenter = track.currentCenter()
                track.bbox = RectF(det.bbox)
                track.missingFrames = 0

                det.trackId = track.id
                matchedDetectionIndices.add(bestMatchIdx)
                matchedTrackIds.add(trackId)
            } else {
                track.missingFrames++
            }
        }

        // 2. Remove dead tracks that exceeded grace period
        activeTracks.entries.removeIf { it.value.missingFrames > maxMissingFrames }

        // 3. Register new tracks for unmatched detections
        for (i in detections.indices) {
            if (!matchedDetectionIndices.contains(i)) {
                val det = detections[i]
                val newId = nextTrackId++
                det.trackId = newId

                activeTracks[newId] = Track(
                    id = newId,
                    bbox = RectF(det.bbox),
                    classId = det.classId,
                    className = det.className
                )
            }
        }

        return detections
    }

    fun getTrack(trackId: Int): Track? = activeTracks[trackId]

    private fun calculateIoU(boxA: RectF, boxB: RectF): Float {
        val interLeft = max(boxA.left, boxB.left)
        val interTop = max(boxA.top, boxB.top)
        val interRight = min(boxA.right, boxB.right)
        val interBottom = min(boxA.bottom, boxB.bottom)

        val interWidth = max(0f, interRight - interLeft)
        val interHeight = max(0f, interBottom - interTop)
        val interArea = interWidth * interHeight

        val areaA = boxA.width() * boxA.height()
        val areaB = boxB.width() * boxB.height()
        val unionArea = areaA + areaB - interArea

        return if (unionArea <= 0f) 0f else interArea / unionArea
    }
}

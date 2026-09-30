package com.nayan.assistive.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * TensorFlow Lite YOLOv8 Object Detector for NAYAN.
 *
 * Configured for 13 classes:
 * 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck,
 * 6: traffic light, 7: fire hydrant, 8: stop sign, 9: bench,
 * 10: chair, 11: stairs, 12: pothole
 */
class YoloDetector(
    private val context: Context,
    private val modelPath: String = "nayan_yolov8n.tflite",
    private val labelPath: String = "labels.txt",
    private val confidenceThreshold: Float = 0.35f,
    private val iouThreshold: Float = 0.45f
) {
    private val interpreter: Interpreter
    val labels: List<String>

    val inputWidth: Int = 640
    val inputHeight: Int = 640

    init {
        val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)
        labels = loadLabels(context, labelPath)
    }

    private fun loadLabels(context: Context, filename: String): List<String> {
        val list = mutableListOf<String>()
        context.assets.open(filename).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                lines.forEach { line ->
                    val trimmed = line.trim()
                    if (trimmed.isNotEmpty()) {
                        list.add(trimmed)
                    }
                }
            }
        }
        return list
    }

    /**
     * Runs inference on the given bitmap and returns bounding boxes in original image dimensions.
     */
    fun detect(bitmap: Bitmap, originalWidth: Int, originalHeight: Int): List<Detection> {
        // 1. Resize input bitmap to 640x640
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)

        // 2. Prepare input buffer (1 * 640 * 640 * 3 * 4 bytes for float32)
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputWidth * inputHeight * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputWidth * inputHeight)
        resized.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        for (pixel in intValues) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        // 3. Prepare output tensor
        // YOLOv8 output shape is typically [1, 4 + numClasses, 8400]
        val numClasses = labels.size
        val channels = 4 + numClasses
        val numAnchors = 8400

        // Handle either [1, channels, 8400] or [1, 8400, channels]
        val outputTensorShape = interpreter.getOutputTensor(0).shape()
        val isChannelsFirst = (outputTensorShape.size == 3 && outputTensorShape[1] == channels)

        val outputArray: Any = if (isChannelsFirst) {
            Array(1) { Array(channels) { FloatArray(numAnchors) } }
        } else {
            Array(1) { Array(numAnchors) { FloatArray(channels) } }
        }

        interpreter.run(inputBuffer, outputArray)

        // 4. Parse candidate detections
        val candidates = mutableListOf<Detection>()

        val scaleX = originalWidth.toFloat() / inputWidth
        val scaleY = originalHeight.toFloat() / inputHeight

        for (anchorIdx in 0 until numAnchors) {
            var cx: Float
            var cy: Float
            var w: Float
            var h: Float

            var bestClassScore = 0f
            var bestClassId = -1

            if (isChannelsFirst) {
                @Suppress("UNCHECKED_CAST")
                val matrix = outputArray as Array<Array<FloatArray>>
                cx = matrix[0][0][anchorIdx]
                cy = matrix[0][1][anchorIdx]
                w = matrix[0][2][anchorIdx]
                h = matrix[0][3][anchorIdx]

                for (c in 0 until numClasses) {
                    val score = matrix[0][4 + c][anchorIdx]
                    if (score > bestClassScore) {
                        bestClassScore = score
                        bestClassId = c
                    }
                }
            } else {
                @Suppress("UNCHECKED_CAST")
                val matrix = outputArray as Array<Array<FloatArray>>
                cx = matrix[0][anchorIdx][0]
                cy = matrix[0][anchorIdx][1]
                w = matrix[0][anchorIdx][2]
                h = matrix[0][anchorIdx][3]

                for (c in 0 until numClasses) {
                    val score = matrix[0][anchorIdx][4 + c]
                    if (score > bestClassScore) {
                        bestClassScore = score
                        bestClassId = c
                    }
                }
            }

            if (bestClassScore >= confidenceThreshold && bestClassId != -1) {
                // Convert center coordinates to frame [left, top, right, bottom]
                val left = max(0f, (cx - (w / 2f)) * scaleX)
                val top = max(0f, (cy - (h / 2f)) * scaleY)
                val right = min(originalWidth.toFloat(), (cx + (w / 2f)) * scaleX)
                val bottom = min(originalHeight.toFloat(), (cy + (h / 2f)) * scaleY)

                val className = if (bestClassId in labels.indices) labels[bestClassId] else "unknown"

                candidates.add(
                    Detection(
                        bbox = RectF(left, top, right, bottom),
                        classId = bestClassId,
                        className = className,
                        confidence = bestClassScore
                    )
                )
            }
        }

        // 5. Apply Non-Maximum Suppression (NMS)
        return applyNMS(candidates, iouThreshold)
    }

    private fun applyNMS(detections: List<Detection>, threshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()

        for (candidate in sorted) {
            var shouldKeep = true
            for (chosen in selected) {
                if (candidate.classId == chosen.classId && calculateIoU(candidate.bbox, chosen.bbox) > threshold) {
                    shouldKeep = false
                    break
                }
            }
            if (shouldKeep) {
                selected.add(candidate)
            }
        }

        return selected
    }

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

    fun close() {
        interpreter.close()
    }
}

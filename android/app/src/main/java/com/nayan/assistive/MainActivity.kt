package com.nayan.assistive

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.nayan.assistive.databinding.ActivityMainBinding
import com.nayan.assistive.ml.YoloDetector
import com.nayan.assistive.priority.HazardPriorityEngine
import com.nayan.assistive.tracking.ObjectTracker
import com.nayan.assistive.tts.TTSManager
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var yoloDetector: YoloDetector
    private lateinit var objectTracker: ObjectTracker
    private lateinit var hazardEngine: HazardPriorityEngine
    private lateinit var ttsManager: TTSManager

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize background executor for ML analysis
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize ML, Tracker, Priority Engine, and TTS
        yoloDetector = YoloDetector(this)
        objectTracker = ObjectTracker()
        hazardEngine = HazardPriorityEngine()
        ttsManager = TTSManager(this)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun allPermissionsGranted(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 1. Camera Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            // 2. Image Analysis Pipeline
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageProxy(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    private fun processImageProxy(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        if (bitmap == null) return

        val frameWidth = bitmap.width
        val frameHeight = bitmap.height

        // 1. Detect objects using TFLite YOLOv8
        val rawDetections = yoloDetector.detect(bitmap, frameWidth, frameHeight)

        // 2. Track objects across frames to preserve unique IDs
        val trackedDetections = objectTracker.update(rawDetections)

        // 3. Score hazards with Path-Awareness & TTC
        val topHazard = hazardEngine.getTopHazard(
            trackedDetections,
            objectTracker,
            frameWidth.toFloat(),
            frameHeight.toFloat()
        )

        // 4. Trigger audio feedback via Text-to-Speech
        ttsManager.processHazardAlert(topHazard)

        // 5. Update UI on Main Thread
        runOnUiThread {
            binding.overlayView.updateResults(
                trackedDetections,
                topHazard,
                frameWidth,
                frameHeight
            )

            if (topHazard != null) {
                val ttcText = topHazard.ttcSeconds?.let { String.format("%.1fs", it) } ?: "Safe"
                val corridorStatus = if (topHazard.pathScore > 0.4f) "IN PATH" else "Side"

                binding.tvTopHazard.text = "${topHazard.detection.className.uppercase()} (Score: ${(topHazard.finalScore * 100).toInt()})"
                binding.tvMetrics.text = "Corridor: $corridorStatus | TTC: $ttcText | Conf: ${(topHazard.detection.confidence * 100).toInt()}%"
            } else {
                binding.tvTopHazard.text = getString(R.string.hazard_status_idle)
                binding.tvMetrics.text = "Corridor: Clear | All directions safe"
            }
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val planes = imageProxy.planes
        if (planes.isEmpty()) return null

        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width

        val bitmap = Bitmap.createBitmap(
            imageProxy.width + rowPadding / pixelStride,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)

        // Apply rotation if needed
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        return if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height, matrix, true)
        } else {
            Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        yoloDetector.close()
        ttsManager.shutdown()
    }
}

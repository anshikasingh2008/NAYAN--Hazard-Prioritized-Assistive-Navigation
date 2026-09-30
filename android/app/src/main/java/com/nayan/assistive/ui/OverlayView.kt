package com.nayan.assistive.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.nayan.assistive.ml.Detection
import com.nayan.assistive.priority.HazardPriorityEngine.HazardScore

/**
 * Custom View for drawing the walking corridor guideline,
 * tracked bounding boxes, and highlighting the prioritized top hazard.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var detections: List<Detection> = emptyList()
    private var topHazard: HazardScore? = null
    private var frameWidth: Int = 1
    private var frameHeight: Int = 1

    private val corridorPaint = Paint().apply {
        color = Color.parseColor("#FFFF00") // Yellow
        strokeWidth = 4f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val corridorFillPaint = Paint().apply {
        color = Color.parseColor("#1AFFFF00") // 10% translucent yellow
        style = Paint.Style.FILL
    }

    private val normalBoxPaint = Paint().apply {
        color = Color.parseColor("#00E676") // Green
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val topHazardBoxPaint = Paint().apply {
        color = Color.parseColor("#FF1744") // Bright Red
        strokeWidth = 8f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    fun updateResults(
        detections: List<Detection>,
        topHazard: HazardScore?,
        frameWidth: Int,
        frameHeight: Int
    ) {
        this.detections = detections
        this.topHazard = topHazard
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width == 0 || height == 0) return

        // 1. Draw central 40% walking corridor (30% to 70% width)
        val corridorLeft = width * 0.30f
        val corridorRight = width * 0.70f

        canvas.drawRect(corridorLeft, 0f, corridorRight, height.toFloat(), corridorFillPaint)
        canvas.drawLine(corridorLeft, 0f, corridorLeft, height.toFloat(), corridorPaint)
        canvas.drawLine(corridorRight, 0f, corridorRight, height.toFloat(), corridorPaint)
        canvas.drawText("WALKING CORRIDOR", corridorLeft + 16f, 60f, textPaint)

        if (frameWidth <= 0 || frameHeight <= 0) return

        val scaleX = width.toFloat() / frameWidth
        val scaleY = height.toFloat() / frameHeight

        // 2. Draw detections
        for (det in detections) {
            val isTop = (topHazard?.detection?.trackId != null && det.trackId == topHazard?.detection?.trackId)
                    || (topHazard?.detection === det)

            val paint = if (isTop) topHazardBoxPaint else normalBoxPaint

            val scaledBox = RectF(
                det.bbox.left * scaleX,
                det.bbox.top * scaleY,
                det.bbox.right * scaleX,
                det.bbox.bottom * scaleY
            )

            canvas.drawRoundRect(scaledBox, 12f, 12f, paint)

            val trackLabel = det.trackId?.let { " [ID: $it]" } ?: ""
            val label = "${det.className}$trackLabel (${(det.confidence * 100).toInt()}%)"
            canvas.drawText(label, scaledBox.left + 8f, maxOf(scaledBox.top - 10f, 40f), textPaint)
        }
    }
}

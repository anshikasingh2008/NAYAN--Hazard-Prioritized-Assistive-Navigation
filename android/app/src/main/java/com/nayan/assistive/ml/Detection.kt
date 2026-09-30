package com.nayan.assistive.ml

import android.graphics.RectF

data class Detection(
    val bbox: RectF,          // [left, top, right, bottom] in frame coordinates
    val classId: Int,
    val className: String,
    val confidence: Float,
    var trackId: Int? = null  // Assigned by ObjectTracker
)

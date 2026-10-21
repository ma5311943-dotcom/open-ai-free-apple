package com.offlineai

import android.content.Intent
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class DownloadNotificationModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "DownloadNotificationModule"

    @ReactMethod
    fun startForegroundService(
        title: String,
        speed: String,
        eta: String,
        downloaded: String,
        total: String,
        left: String,
        percent: Int,
        promise: Promise
    ) {
        try {
            val intent = Intent(reactContext, DownloadNotificationService::class.java).apply {
                action = DownloadNotificationService.ACTION_START
                putExtra(DownloadNotificationService.EXTRA_TITLE, title)
                putExtra(DownloadNotificationService.EXTRA_SPEED, speed)
                putExtra(DownloadNotificationService.EXTRA_ETA, eta)
                putExtra(DownloadNotificationService.EXTRA_DOWNLOADED, downloaded)
                putExtra(DownloadNotificationService.EXTRA_TOTAL, total)
                putExtra(DownloadNotificationService.EXTRA_LEFT, left)
                putExtra(DownloadNotificationService.EXTRA_PERCENT, percent)
            }

            ContextCompat.startForegroundService(reactContext, intent)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("START_SERVICE_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun updateProgress(
        title: String,
        speed: String,
        eta: String,
        downloaded: String,
        total: String,
        left: String,
        percent: Int
    ) {
        try {
            val intent = Intent(reactContext, DownloadNotificationService::class.java).apply {
                action = DownloadNotificationService.ACTION_UPDATE
                putExtra(DownloadNotificationService.EXTRA_TITLE, title)
                putExtra(DownloadNotificationService.EXTRA_SPEED, speed)
                putExtra(DownloadNotificationService.EXTRA_ETA, eta)
                putExtra(DownloadNotificationService.EXTRA_DOWNLOADED, downloaded)
                putExtra(DownloadNotificationService.EXTRA_TOTAL, total)
                putExtra(DownloadNotificationService.EXTRA_LEFT, left)
                putExtra(DownloadNotificationService.EXTRA_PERCENT, percent)
            }
            reactContext.startService(intent)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    @ReactMethod
    fun stopForegroundService(promise: Promise) {
        try {
            val intent = Intent(reactContext, DownloadNotificationService::class.java).apply {
                action = DownloadNotificationService.ACTION_STOP
            }
            reactContext.startService(intent)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("STOP_SERVICE_ERROR", e.message, e)
        }
    }
}

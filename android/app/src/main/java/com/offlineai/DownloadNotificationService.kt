package com.offlineai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

class DownloadNotificationService : Service() {

    companion object {
        const val CHANNEL_ID = "offline_ai_download_channel"
        const val CHANNEL_NAME = "Model Downloads"
        const val NOTIFICATION_ID = 9901

        const val ACTION_START = "com.offlineai.download.START"
        const val ACTION_UPDATE = "com.offlineai.download.UPDATE"
        const val ACTION_STOP = "com.offlineai.download.STOP"

        const val EXTRA_TITLE = "extra_title"
        const val EXTRA_SPEED = "extra_speed"
        const val EXTRA_ETA = "extra_eta"
        const val EXTRA_DOWNLOADED = "extra_downloaded"
        const val EXTRA_TOTAL = "extra_total"
        const val EXTRA_LEFT = "extra_left"
        const val EXTRA_PERCENT = "extra_percent"
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null
    private var notificationManager: NotificationManager? = null

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
        createNotificationChannel()
        acquireLocks()
    }

    private fun acquireLocks() {
        try {
            val powerManager = getSystemService(Context.POWER_SERVICE) as? PowerManager
            if (wakeLock == null && powerManager != null) {
                wakeLock = powerManager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "OfflineAI:DownloadWakeLock"
                ).apply {
                    setReferenceCounted(false)
                    acquire(24 * 60 * 60 * 1000L) // 24 hours max safeguard
                }
            }

            val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            if (wifiLock == null && wifiManager != null) {
                wifiLock = wifiManager.createWifiLock(
                    WifiManager.WIFI_MODE_FULL_HIGH_PERF,
                    "OfflineAI:DownloadWifiLock"
                ).apply {
                    setReferenceCounted(false)
                    acquire()
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun releaseLocks() {
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (_: Exception) {}
        wakeLock = null

        try {
            if (wifiLock?.isHeld == true) {
                wifiLock?.release()
            }
        } catch (_: Exception) {}
        wifiLock = null
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows progress, speed, and time remaining for offline AI model downloads"
                setShowBadge(false)
                setSound(null, null)
                enableVibration(false)
            }
            notificationManager?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(
        title: String,
        speed: String,
        eta: String,
        downloaded: String,
        total: String,
        left: String,
        percent: Int
    ): Notification {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }

        val pendingIntent = if (launchIntent != null) {
            val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
            PendingIntent.getActivity(this, 0, launchIntent, flags)
        } else null

        val speedEtaText = when {
            speed.isNotEmpty() && eta.isNotEmpty() -> "⚡ $speed   •   ⏳ $eta"
            speed.isNotEmpty() -> "⚡ $speed"
            eta.isNotEmpty() -> "⏳ $eta"
            else -> "Downloading model in background..."
        }

        val sizeText = when {
            downloaded.isNotEmpty() && total.isNotEmpty() && left.isNotEmpty() ->
                "💾 $downloaded / $total ($left left)"
            downloaded.isNotEmpty() && total.isNotEmpty() ->
                "💾 $downloaded / $total"
            else -> ""
        }

        val fullTitle = if (percent in 1..99) "$title ($percent%)" else title

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(fullTitle)
            .setContentText(speedEtaText)
            .setStyle(
                NotificationCompat.BigTextStyle()
                    .setSummaryText(if (percent > 0) "$percent%" else null)
                    .bigText(if (sizeText.isNotEmpty()) "$speedEtaText\n$sizeText" else speedEtaText)
            )
            .setProgress(100, percent.coerceIn(0, 100), percent <= 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)

        if (sizeText.isNotEmpty()) {
            builder.setSubText(sizeText)
        }

        if (pendingIntent != null) {
            builder.setContentIntent(pendingIntent)
        }

        return builder.build()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) return START_NOT_STICKY

        when (intent.action) {
            ACTION_START -> {
                acquireLocks()
                val title = intent.getStringExtra(EXTRA_TITLE) ?: "Downloading AI Model"
                val speed = intent.getStringExtra(EXTRA_SPEED) ?: ""
                val eta = intent.getStringExtra(EXTRA_ETA) ?: ""
                val downloaded = intent.getStringExtra(EXTRA_DOWNLOADED) ?: ""
                val total = intent.getStringExtra(EXTRA_TOTAL) ?: ""
                val left = intent.getStringExtra(EXTRA_LEFT) ?: ""
                val percent = intent.getIntExtra(EXTRA_PERCENT, 0)

                val notification = buildNotification(title, speed, eta, downloaded, total, left, percent)

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    startForeground(
                        NOTIFICATION_ID,
                        notification,
                        ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                    )
                } else {
                    startForeground(NOTIFICATION_ID, notification)
                }
            }

            ACTION_UPDATE -> {
                val title = intent.getStringExtra(EXTRA_TITLE) ?: "Downloading AI Model"
                val speed = intent.getStringExtra(EXTRA_SPEED) ?: ""
                val eta = intent.getStringExtra(EXTRA_ETA) ?: ""
                val downloaded = intent.getStringExtra(EXTRA_DOWNLOADED) ?: ""
                val total = intent.getStringExtra(EXTRA_TOTAL) ?: ""
                val left = intent.getStringExtra(EXTRA_LEFT) ?: ""
                val percent = intent.getIntExtra(EXTRA_PERCENT, 0)

                val notification = buildNotification(title, speed, eta, downloaded, total, left, percent)
                notificationManager?.notify(NOTIFICATION_ID, notification)
            }

            ACTION_STOP -> {
                releaseLocks()
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                } else {
                    @Suppress("DEPRECATION")
                    stopForeground(true)
                }
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseLocks()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}

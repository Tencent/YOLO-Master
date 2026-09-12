package dev.yolomaster.app

import android.app.Application
import dev.yolomaster.app.model.ModelCatalog
import dev.yolomaster.app.ui.bench.BenchHistory

/**
 * Process-wide singletons live here so every tab shares one model catalog (the iOS app
 * discovers bundled models per tab; on Android the asset copy is a one-time cost we do once).
 */
class YoloMasterApp : Application() {
    lateinit var catalog: ModelCatalog
        private set
    lateinit var history: BenchHistory
        private set

    override fun onCreate() {
        super.onCreate()
        catalog = ModelCatalog(this)
        history = BenchHistory(this)
    }

    companion object {
        fun from(ctx: android.content.Context): YoloMasterApp = ctx.applicationContext as YoloMasterApp
    }
}

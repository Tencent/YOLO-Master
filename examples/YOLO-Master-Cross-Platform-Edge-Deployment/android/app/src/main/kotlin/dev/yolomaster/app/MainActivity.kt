package dev.yolomaster.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import dev.yolomaster.app.ui.RootScreen
import dev.yolomaster.app.ui.theme.YoloMasterTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge-to-edge like the iOS app: the camera preview runs under the status bar and the
        // floating cards sit in the safe-area insets.
        enableEdgeToEdge()
        setContent {
            YoloMasterTheme {
                RootScreen()
            }
        }
    }
}

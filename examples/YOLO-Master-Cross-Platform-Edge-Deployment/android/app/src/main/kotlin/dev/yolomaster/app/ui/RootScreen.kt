package dev.yolomaster.app.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CenterFocusWeak
import androidx.compose.material.icons.filled.Photo
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import dev.yolomaster.app.ui.bench.BenchScreen
import dev.yolomaster.app.ui.live.LiveScreen
import dev.yolomaster.app.ui.photo.PhotoScreen
import dev.yolomaster.app.ui.settings.SettingsScreen

/** The iOS `TabView`: Live, Photo, Bench, Settings in this order; Live selected at launch. */
enum class Tab(val label: String, val icon: ImageVector) {
    Live("Live", Icons.Filled.CenterFocusWeak),
    Photo("Photo", Icons.Filled.Photo),
    Bench("Bench", Icons.Filled.Speed),
    Settings("Settings", Icons.Filled.Settings),
}

@Composable
fun RootScreen() {
    var tabIndex by rememberSaveable { mutableIntStateOf(0) }
    val tab = Tab.entries[tabIndex]
    Scaffold(
        bottomBar = {
            NavigationBar {
                Tab.entries.forEachIndexed { i, t ->
                    NavigationBarItem(
                        selected = i == tabIndex,
                        onClick = { tabIndex = i },
                        icon = { Icon(t.icon, contentDescription = t.label) },
                        label = { Text(t.label) },
                    )
                }
            }
        },
    ) { inner ->
        // Screens are composed conditionally so leaving a tab disposes it (camera, runtime,
        // bench loop all stop), matching the iOS onDisappear behaviour.
        Box(Modifier.fillMaxSize().padding(bottom = inner.calculateBottomPadding())) {
            when (tab) {
                Tab.Live -> LiveScreen()
                Tab.Photo -> PhotoScreen()
                Tab.Bench -> BenchScreen()
                Tab.Settings -> SettingsScreen()
            }
        }
    }
}

/** Temporary stand-in used by tabs not built yet. */
@Composable
internal fun Placeholder(name: String) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text(name) }
}

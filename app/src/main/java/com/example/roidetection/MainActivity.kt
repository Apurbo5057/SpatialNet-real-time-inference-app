package com.example.roidetection

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.roidetection.ui.HomeScreen
import com.example.roidetection.ui.InferenceScreen
import com.example.roidetection.ui.ObjectListScreen
import com.example.roidetection.ui.theme.ROIDetectionTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        androidx.core.content.res.ResourcesCompat.getFont(this, R.font.atkinson_bold)
            ?.let { com.example.roidetection.ui.OverlayFont.typeface = it }
        setContent {
            ROIDetectionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ROIDetectionApp()
                }
            }
        }
    }
}

@Composable
fun ROIDetectionApp() {
    val navController = rememberNavController()
    val viewModel: InferenceViewModel = viewModel()
    val showDetails by viewModel.showDetails.collectAsState()

    NavHost(
        navController = navController,
        startDestination = "home"
    ) {
        composable("home") {
            HomeScreen(
                onOpenMode = { mode -> navController.navigate("mode/${mode.name}") },
                showDetails = showDetails,
                onShowDetailsChange = viewModel::setShowDetails
            )
        }
        composable("mode/{mode}") { entry ->
            val mode = entry.arguments?.getString("mode")
                ?.let { name -> AppMode.entries.firstOrNull { it.name == name } }
                ?: AppMode.IDENTIFY
            InferenceScreen(
                viewModel = viewModel,
                mode = mode,
                onBack = { navController.popBackStack() },
                onOpenObjectList = { navController.navigate("objects") }
            )
        }
        composable("objects") {
            ObjectListScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }
    }
}

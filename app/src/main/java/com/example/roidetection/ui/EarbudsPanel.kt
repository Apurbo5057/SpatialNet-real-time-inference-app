package com.example.roidetection.ui

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.content.Intent
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import kotlinx.coroutines.delay
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.roidetection.CombinedResult
import com.example.roidetection.InferenceViewModel
import com.example.roidetection.earbuds.EarbudModel
import com.example.roidetection.earbuds.PairState
import com.example.roidetection.ui.theme.Beam
import com.example.roidetection.ui.theme.Coral
import com.example.roidetection.ui.theme.GlassLine
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.MaterialTheme


/**
 * Pair Earbuds panel: the caption for the pointed-at earbuds, their details, and a
 * Connect button that walks through Android's pairing flow with plain-language status.
 */
@Composable
fun EarbudsPanel(result: CombinedResult, viewModel: InferenceViewModel, speaker: Speaker) {
    val context = LocalContext.current
    val pairState by viewModel.pairer.state.collectAsState()
    val ready by viewModel.earbudsReady.collectAsState()
    val error by viewModel.earbudsError.collectAsState()

    val match = result.earbudsMatch
    val model = match?.model
    // What Connect searches for, frozen when tapped so a flickering caption can't change it.
    var targets by remember { mutableStateOf<List<EarbudModel>>(emptyList()) }
    var showSteps by remember { mutableStateOf(false) }
    var showDetails by remember { mutableStateOf<EarbudModel?>(null) }
    var showSupported by remember { mutableStateOf(false) }

    val chooser = rememberLauncherForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) {
        viewModel.pairer.onChooserResult(it.resultCode == android.app.Activity.RESULT_OK, it.data)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) viewModel.pairer.retry()
    }
    val enableBluetooth = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        viewModel.pairer.retry()
    }
    LaunchedEffect(pairState) {
        when (val st = pairState) {
            is PairState.ChooserReady -> {
                chooser.launch(IntentSenderRequest.Builder(st.sender).build())
                viewModel.pairer.onChooserShown()
            }
            // Success clears itself: no status is left on screen after pairing.
            is PairState.Paired -> when {
                st.connected -> { delay(4_000); viewModel.pairer.reset() }
                st.audioChecked -> { delay(10_000); viewModel.pairer.reset() }
                else -> Unit
            }
            is PairState.AlreadyPaired -> { delay(6_000); viewModel.pairer.reset() }
            else -> Unit
        }
    }
    val openBluetoothSettings = { context.startActivity(Intent(Settings.ACTION_BLUETOOTH_SETTINGS)) }

    GlassPanel {
            // Caption row: same text as the label on the box.
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                val label = result.roiLabel
                AnswerText(
                    text = when {
                        error != null -> error!!
                        !ready -> "Getting the earbuds list ready…"
                        label != null -> label.text
                        else -> "Point at earbuds or their case"
                    },
                    confident = label?.confident == true,
                    isHint = label == null,
                    modifier = Modifier.weight(1f)
                )
                GlassIconButton(AppIcons.Speaker, "Say it", onClick = { label?.let { speaker.speak(it.spoken) } }, enabled = label != null)
                GlassIconButton(Icons.Filled.Info, "Details", onClick = { showDetails = model }, enabled = model != null)
            }

            if (pairState == PairState.Idle) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    BeamButton("Connect", enabled = model != null, icon = AppIcons.Bluetooth, modifier = Modifier.weight(1f), onClick = {
                        viewModel.pairer.start(match?.candidates?.ifEmpty { null } ?: listOfNotNull(model))
                    })
                    GlassButton("Supported", enabled = ready, icon = Icons.Filled.List, modifier = Modifier.weight(1f),
                        onClick = { showSupported = true })
                }
            } else {
                HorizontalDivider(color = GlassLine)
                PairStatus(
                    state = pairState,
                    onAllow = { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) permission.launch(Manifest.permission.BLUETOOTH_CONNECT) },
                    onTurnOn = { enableBluetooth.launch(Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)) },
                    onRetry = { viewModel.pairer.retry() },
                    onShowSteps = { targets = viewModel.pairer.currentTargets; showSteps = true },
                    onSearchAll = { viewModel.pairer.searchAgain(searchAll = true) },
                    onPairAgain = { viewModel.pairer.searchAgain(searchAll = false) },
                    onBluetoothSettings = openBluetoothSettings,
                    onDone = { viewModel.pairer.reset() }
                )
            }
    }

    if (showSteps && targets.isNotEmpty()) {
        PairingStepsDialog(targets = targets, onClose = { showSteps = false })
    }
    showDetails?.let { m ->
        EarbudsDetailsDialog(
            model = m,
            thumbnail = remember(m.id) { viewModel.earbudsThumbnail(m) },
            onConnect = {
                showDetails = null
                viewModel.pairer.start(listOf(m) + (match?.candidates?.filter { it.id != m.id } ?: emptyList()))
            },
            onClose = { showDetails = null }
        )
    }
    if (showSupported) {
        SupportedEarbudsDialog(viewModel, onPick = { showSupported = false; showDetails = it }, onClose = { showSupported = false })
    }
}

@Composable
private fun PairStatus(
    state: PairState,
    onAllow: () -> Unit,
    onTurnOn: () -> Unit,
    onRetry: () -> Unit,
    onShowSteps: () -> Unit,
    onSearchAll: () -> Unit,
    onPairAgain: () -> Unit,
    onBluetoothSettings: () -> Unit,
    onDone: () -> Unit
) {
    @Composable
    fun Line(text: String, color: Color = Color.White, busy: Boolean = false) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (busy) {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = Beam)
                Spacer(Modifier.size(12.dp))
            }
            Text(text, color = color, style = MaterialTheme.typography.bodyLarge)
        }
    }

    @Composable
    fun Actions(vararg actions: Pair<String, () -> Unit>) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            actions.forEachIndexed { i, (text, onClick) ->
                if (i == 0) BeamButton(text, onClick = onClick, modifier = Modifier.weight(1f))
                else TextButton(onClick = onClick) { Text(text, color = Color.White, style = MaterialTheme.typography.labelLarge) }
            }
        }
    }

    when (state) {
        PairState.Idle -> Unit
        PairState.NeedsPermission -> {
            Line("To pair, allow the app to find and connect to nearby devices.")
            Actions("Allow" to onAllow, "Cancel" to onDone)
        }
        PairState.BluetoothOff -> {
            Line("Bluetooth is off.")
            Actions("Turn on Bluetooth" to onTurnOn, "Cancel" to onDone)
        }
        PairState.Unsupported -> {
            Line("This Android version can't pair from the app. Pair in Bluetooth settings instead.")
            Actions("Bluetooth settings" to onBluetoothSettings, "Close" to onDone)
        }
        is PairState.AlreadyPaired -> {
            Line(
                if (state.connected) "✓ ${state.deviceName} is already paired and connected."
                else "✓ ${state.deviceName} is already paired with this phone. Open Bluetooth settings and tap it to connect.",
                Beam
            )
            Actions("Done" to onDone, "Bluetooth settings" to onBluetoothSettings, "Pair again" to onPairAgain)
        }
        is PairState.ChooserReady, is PairState.Searching -> {
            val names = (state as? PairState.Searching)?.targets ?: emptyList()
            Line(
                "Looking for ${names.joinToString(" or ").ifEmpty { "your earbuds" }}. Put them in pairing mode: " +
                        "usually open the case and hold its button until the light blinks.",
                busy = true
            )
            Actions("How to" to onShowSteps, "Cancel" to onDone)
        }
        is PairState.Pairing -> {
            Line("Pairing with ${state.deviceName}… Accept any pairing request Android shows.", busy = true)
        }
        is PairState.Paired -> {
            Line(
                when {
                    state.connected -> "✓ Connected to ${state.deviceName}."
                    !state.audioChecked -> "✓ Paired with ${state.deviceName}."
                    else -> "✓ Paired with ${state.deviceName}. If no sound plays, tap it in Bluetooth settings."
                },
                Beam
            )
            if (state.audioChecked && !state.connected) Actions("Done" to onDone, "Bluetooth settings" to onBluetoothSettings)
            else Actions("Done" to onDone)
        }
        is PairState.NotFound -> {
            Line(
                if (state.searchedAll) "No earbuds were chosen. Make sure they are in pairing mode, then try again."
                else "Couldn't find ${state.targets.joinToString(" or ")}. Make sure they are in pairing mode " +
                        "(light blinking) and near the phone."
            )
            if (state.searchedAll) Actions("Try again" to onRetry, "Close" to onDone)
            else Actions("Try again" to onRetry, "Show all devices" to onSearchAll, "Close" to onDone)
        }
        is PairState.Failed -> {
            Line(state.message, Coral)
            Actions("Try again" to onRetry, "Close" to onDone)
        }
    }
}

/** The model's own pairing-mode steps, while the search runs. */
@Composable
private fun PairingStepsDialog(targets: List<EarbudModel>, onClose: () -> Unit) {
    val main = targets.first()
    AlertDialog(
        onDismissRequest = onClose,
        title = { Text("Get your ${main.displayName} ready") },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Put the earbuds in pairing mode:", fontWeight = FontWeight.SemiBold)
                main.pairingSteps.forEachIndexed { i, step -> Text("${i + 1}. $step", fontSize = 15.sp) }
                if (targets.size > 1) {
                    Text(
                        "The app will also look for ${targets.drop(1).joinToString(" and ") { it.displayName }}, " +
                                "in case it guessed wrong.",
                        fontSize = 13.sp
                    )
                }
                Text("The app is already searching. Android will ask you to allow the connection.", fontSize = 13.sp)
            }
        },
        confirmButton = { Button(onClick = onClose) { Text("OK") } }
    )
}

/** Photo, specs and features for one model. */
@Composable
private fun EarbudsDetailsDialog(
    model: EarbudModel,
    thumbnail: android.graphics.Bitmap?,
    onConnect: () -> Unit,
    onClose: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onClose,
        title = { Text(model.displayName) },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                thumbnail?.let {
                    Image(
                        it.asImageBitmap(), contentDescription = model.displayName,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.size(140.dp).clip(RoundedCornerShape(12.dp)).align(Alignment.CenterHorizontally)
                    )
                }
                model.specs.forEach { (label, value) ->
                    Row {
                        Text(label, fontWeight = FontWeight.SemiBold, fontSize = 14.sp, modifier = Modifier.weight(0.42f))
                        Text(value, fontSize = 14.sp, modifier = Modifier.weight(0.58f))
                    }
                }
                if (model.features.isNotEmpty()) {
                    Text("Features", fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                    model.features.take(8).forEach { Text("• $it", fontSize = 13.sp) }
                }
                model.defaultNames.firstOrNull()?.let {
                    Text(
                        "Bluetooth name: $it" + if (model.verifiedOnDevice) "" else " (not yet checked on real earbuds)",
                        fontSize = 12.sp
                    )
                }
            }
        },
        confirmButton = { Button(onClick = onConnect) { Text("🔗 Connect") } },
        dismissButton = { TextButton(onClick = onClose) { Text("Close") } }
    )
}

@Composable
private fun SupportedEarbudsDialog(viewModel: InferenceViewModel, onPick: (EarbudModel) -> Unit, onClose: () -> Unit) {
    val models = remember { viewModel.supportedEarbuds() }
    AlertDialog(
        onDismissRequest = onClose,
        title = { Text("Supported earbuds") },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                models.forEach { m ->
                    val thumb = remember(m.id) { viewModel.earbudsThumbnail(m) }
                    TextButton(onClick = { onPick(m) }, contentPadding = PaddingValues(0.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                            thumb?.let {
                                Image(it.asImageBitmap(), contentDescription = null, contentScale = ContentScale.Crop,
                                    modifier = Modifier.size(48.dp).clip(RoundedCornerShape(8.dp)))
                            }
                            Spacer(Modifier.size(12.dp))
                            Text(m.displayName, fontSize = 16.sp)
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onClose) { Text("Close") } }
    )
}

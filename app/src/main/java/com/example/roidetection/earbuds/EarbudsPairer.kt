package com.example.roidetection.earbuds

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanResult
import android.companion.AssociationInfo
import android.companion.AssociationRequest
import android.companion.BluetoothDeviceFilter
import android.companion.CompanionDeviceManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.IntentSender
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.Parcelable
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.regex.Pattern

/** Where pairing is, in words the Earbuds panel can show. */
sealed interface PairState {
    data object Idle : PairState
    /** Android 12+ needs the "Nearby devices" permission before anything else. */
    data object NeedsPermission : PairState
    data object BluetoothOff : PairState
    /** Android 7 has no CompanionDeviceManager; the user pairs in Bluetooth settings. */
    data object Unsupported : PairState
    /** Already paired with this phone; [connected] if it is playing audio now. */
    data class AlreadyPaired(val deviceName: String, val connected: Boolean) : PairState
    /** Android's device chooser is ready to be shown (the UI launches [sender]). */
    data class ChooserReady(val sender: IntentSender) : PairState
    /** Android is scanning for matching earbuds. */
    data class Searching(val targets: List<String>) : PairState
    data class Pairing(val deviceName: String) : PairState
    /**
     * Paired. [connected] once Android has connected audio (it usually does by itself);
     * [audioChecked] once the app stopped waiting for that.
     */
    data class Paired(val deviceName: String, val connected: Boolean, val audioChecked: Boolean = false) : PairState
    /** Nothing matching was found, or the user cancelled the chooser. */
    data class NotFound(val targets: List<String>, val searchedAll: Boolean) : PairState
    data class Failed(val message: String) : PairState
}

/**
 * Pairs the phone with earbuds through Android's CompanionDeviceManager: Android scans
 * for devices whose Bluetooth name matches the candidate models, shows its own chooser
 * ("Allow this app to connect to realme Buds Air7?"), and returns the chosen device,
 * which is then bonded with [BluetoothDevice.createBond]. No location permission is needed.
 */
@SuppressLint("MissingPermission") // every Bluetooth call is behind hasConnectPermission()
class EarbudsPairer(context: Context) {

    companion object {
        private const val TAG = "EarbudsPairer"
    }

    private val app = context.applicationContext
    private val adapter: BluetoothAdapter? = app.getSystemService(BluetoothManager::class.java)?.adapter
    private val main = Handler(Looper.getMainLooper())

    private val _state = MutableStateFlow<PairState>(PairState.Idle)
    val state: StateFlow<PairState> = _state.asStateFlow()

    private var targets: List<EarbudModel> = emptyList()
    private var lastSearchAll = false
    /** The device being bonded; null once its result has been handled. */
    private var bonding: BluetoothDevice? = null
    private var bondReceiver: BroadcastReceiver? = null

    fun hasConnectPermission(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
                ContextCompat.checkSelfPermission(app, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED

    fun reset() {
        finishBonding()
        _state.value = PairState.Idle
    }

    /**
     * Starts pairing with whichever of [candidates] is nearby in pairing mode. With
     * [searchAll] the chooser lists every nearby Bluetooth device instead, for earbuds
     * whose real Bluetooth name differs from the catalog.
     */
    fun start(candidates: List<EarbudModel>, searchAll: Boolean = false) {
        targets = candidates
        finishBonding()
        when {
            Build.VERSION.SDK_INT < Build.VERSION_CODES.O -> { _state.value = PairState.Unsupported; return }
            !hasConnectPermission() -> { _state.value = PairState.NeedsPermission; return }
            adapter == null -> { _state.value = PairState.Failed("This phone has no Bluetooth."); return }
            !adapter.isEnabled -> { _state.value = PairState.BluetoothOff; return }
        }

        // Already paired? Then there is nothing to search for.
        if (!searchAll) {
            adapter!!.bondedDevices.firstOrNull { d -> candidates.any { it.matchesName(d.name) } }?.let { d ->
                Log.i(TAG, "Already paired: ${d.name}")
                _state.value = PairState.AlreadyPaired(d.name ?: "earbuds", false)
                checkAudioConnection(d) { connected ->
                    if (_state.value is PairState.AlreadyPaired) _state.value = PairState.AlreadyPaired(d.name ?: "earbuds", connected)
                }
                return
            }
        }
        associate(candidates, searchAll)
    }

    /** The models the current or last search was for. */
    val currentTargets: List<EarbudModel> get() = targets

    /** Starts the last search again (after a permission prompt, turning Bluetooth on, or a failure). */
    fun retry() = start(targets, lastSearchAll)

    /** Pair anyway, even though a matching device is already paired. */
    fun searchAgain(searchAll: Boolean) = associate(targets, searchAll)

    private fun associate(candidates: List<EarbudModel>, searchAll: Boolean) {
        val cdm = app.getSystemService(CompanionDeviceManager::class.java)
        if (cdm == null) {
            _state.value = PairState.Unsupported
            return
        }
        val builder = AssociationRequest.Builder().setSingleDevice(false)
        if (searchAll) {
            builder.addDeviceFilter(BluetoothDeviceFilter.Builder().build())
        } else {
            candidates.flatMap { it.namePatterns }.distinct().forEach { pattern ->
                runCatching { Pattern.compile(pattern) }.getOrNull()?.let {
                    builder.addDeviceFilter(BluetoothDeviceFilter.Builder().setNamePattern(it).build())
                }
            }
        }
        val names = candidates.map { it.displayName }
        lastSearchAll = searchAll
        _state.value = PairState.Searching(names)

        val callback = object : CompanionDeviceManager.Callback() {
            // Android 13+
            override fun onAssociationPending(intentSender: IntentSender) {
                _state.value = PairState.ChooserReady(intentSender)
            }

            @Deprecated("Replaced by onAssociationPending on Android 13+")
            override fun onDeviceFound(intentSender: IntentSender) {
                _state.value = PairState.ChooserReady(intentSender)
            }

            override fun onFailure(error: CharSequence?) {
                Log.w(TAG, "Association failed: $error")
                _state.value = PairState.NotFound(names, searchAll)
            }
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                cdm.associate(builder.build(), app.mainExecutor, callback)
            } else {
                @Suppress("DEPRECATION")
                cdm.associate(builder.build(), callback, main)
            }
        } catch (e: Exception) {
            Log.e(TAG, "associate() failed", e)
            _state.value = PairState.Failed("Could not start the Bluetooth search: ${e.message}")
        }
    }

    /** The UI launched the chooser; go back to "searching" so it is not launched twice. */
    fun onChooserShown() {
        if (_state.value is PairState.ChooserReady) _state.value = PairState.Searching(targets.map { it.displayName })
    }

    /** The UI calls this with the chooser's result. */
    fun onChooserResult(ok: Boolean, data: Intent?) {
        val device = if (ok) deviceFrom(data) else null
        if (device == null) {
            _state.value = PairState.NotFound(targets.map { it.displayName }, lastSearchAll)
            return
        }
        bond(device)
    }

    private fun deviceFrom(data: Intent?): BluetoothDevice? {
        data ?: return null
        @Suppress("DEPRECATION")
        when (val extra = data.getParcelableExtra<Parcelable>(CompanionDeviceManager.EXTRA_DEVICE)) {
            is BluetoothDevice -> return extra
            is ScanResult -> return extra.device
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val info = data.getParcelableExtra(CompanionDeviceManager.EXTRA_ASSOCIATION, AssociationInfo::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                info?.associatedDevice?.bluetoothDevice?.let { return it }
                info?.associatedDevice?.bleDevice?.device?.let { return it }
            }
            info?.deviceMacAddress?.toString()?.uppercase()?.let { mac -> return adapter?.getRemoteDevice(mac) }
        }
        return null
    }

    private fun bond(device: BluetoothDevice) {
        val name = device.name ?: device.address
        Log.i(TAG, "Chosen device: $name (${device.address}), bond state ${device.bondState}")
        if (device.bondState == BluetoothDevice.BOND_BONDED) {
            onBonded(device)
            return
        }
        _state.value = PairState.Pairing(name)
        bonding = device
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val d: BluetoothDevice? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE, BluetoothDevice::class.java)
                } else {
                    @Suppress("DEPRECATION") intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                }
                if (d?.address != device.address) return
                val now = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.ERROR)
                val before = intent.getIntExtra(BluetoothDevice.EXTRA_PREVIOUS_BOND_STATE, BluetoothDevice.ERROR)
                Log.i(TAG, "Bond broadcast $before -> $now for $name")
                onBondState(device, now, sawBonding = before == BluetoothDevice.BOND_BONDING)
            }
        }
        bondReceiver = receiver
        // Bond-state broadcasts come from Android's Bluetooth app, a separate app on newer
        // Android, so a not-exported receiver never hears them. Exported, and checked below.
        ContextCompat.registerReceiver(app, receiver, IntentFilter(BluetoothDevice.ACTION_BOND_STATE_CHANGED),
            ContextCompat.RECEIVER_EXPORTED)
        // createBond() returns false when pairing is already under way (e.g. Google Fast Pair
        // started it); the poll below still sees it finish.
        val started = device.createBond()
        Log.i(TAG, "createBond() -> $started")
        pollBond(device, System.currentTimeMillis(), sawBonding = started)
    }

    /** Backup for a missed broadcast: checks the bond state twice a second, for up to 45 s. */
    private fun pollBond(device: BluetoothDevice, startMs: Long, sawBonding: Boolean) {
        main.postDelayed({
            if (bonding?.address != device.address) return@postDelayed
            val now = device.bondState
            val seen = sawBonding || now == BluetoothDevice.BOND_BONDING
            when {
                now == BluetoothDevice.BOND_BONDED -> onBondState(device, now, seen)
                now == BluetoothDevice.BOND_NONE && seen -> onBondState(device, now, true)
                System.currentTimeMillis() - startMs > 45_000 -> {
                    finishBonding()
                    _state.value = PairState.Failed("Pairing with ${device.name ?: "the earbuds"} took too long. " +
                            "Put them back in pairing mode and try again.")
                }
                now == BluetoothDevice.BOND_NONE && System.currentTimeMillis() - startMs > 8_000 -> {
                    finishBonding()
                    _state.value = PairState.Failed("Android did not start pairing. Put the earbuds in pairing mode and try again.")
                }
                else -> pollBond(device, startMs, seen)
            }
        }, 500)
    }

    private fun onBondState(device: BluetoothDevice, now: Int, sawBonding: Boolean) {
        if (bonding?.address != device.address) return  // already handled
        when {
            now == BluetoothDevice.BOND_BONDED -> { finishBonding(); onBonded(device) }
            now == BluetoothDevice.BOND_NONE && sawBonding -> {
                finishBonding()
                _state.value = PairState.Failed("Pairing with ${device.name ?: "the earbuds"} did not finish. " +
                        "Put them back in pairing mode and try again.")
            }
        }
    }

    private fun finishBonding() {
        bonding = null
        unregisterBondReceiver()
    }

    private fun onBonded(device: BluetoothDevice) {
        val name = device.name ?: device.address
        Log.i(TAG, "Paired with $name")
        _state.value = PairState.Paired(name, connected = false)
        checkAudioConnection(device) { connected ->
            if ((_state.value as? PairState.Paired)?.deviceName == name) {
                _state.value = PairState.Paired(name, connected, audioChecked = true)
            }
        }
    }

    /**
     * Android normally connects music and calls by itself right after pairing; this
     * watches for that for up to ~10 s. Apps cannot force the audio connection.
     */
    private fun checkAudioConnection(device: BluetoothDevice, onResult: (Boolean) -> Unit) {
        val a = adapter ?: return onResult(false)
        a.getProfileProxy(app, object : BluetoothProfile.ServiceListener {
            override fun onServiceConnected(profile: Int, proxy: BluetoothProfile) {
                var tries = 0
                val poll = object : Runnable {
                    override fun run() {
                        val connected = proxy.getConnectionState(device) == BluetoothProfile.STATE_CONNECTED
                        if (connected || ++tries >= 10) {
                            onResult(connected)
                            a.closeProfileProxy(profile, proxy)
                        } else {
                            main.postDelayed(this, 1000)
                        }
                    }
                }
                main.post(poll)
            }

            override fun onServiceDisconnected(profile: Int) = Unit
        }, BluetoothProfile.A2DP)
    }

    private fun unregisterBondReceiver() {
        bondReceiver?.let { runCatching { app.unregisterReceiver(it) } }
        bondReceiver = null
    }
}

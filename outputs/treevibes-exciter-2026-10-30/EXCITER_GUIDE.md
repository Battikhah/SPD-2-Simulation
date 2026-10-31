# TreeVibes lab audio → exciter → project piezo

Prepared 30 October 2026. This is a mechanical replay bench protocol. Physical operation has not been tested here.

**Current activity-only bench files:** use `playback_48k/silence_10s.wav` for the silence control, then a short event from `rpw_bursts/playback_48k/`. `rpw_bursts/playback_48k/playlist.wav` is the new burst audition sequence; its gaps are artificial. This supersedes the earlier full-length `playback_48k/candidate_01.wav` examples for the user's requested RPW-assumed activity playback. The burst manifest preserves source timestamps and gains. Full sources remain under `original_8k/infested/`, older full-length derivatives under `bandpassed_8k/` and `attenuated_8k/`, and clean controls under `original_8k/clean/`.

## What the audio means

Use the package manifest to trace each clip to TreeVibes v1. The lab README says recordings were made from logs in the lab with a Roland R4 and divided from long recordings into small files. A file in `infested` is a candidate borer-activity recording. Available metadata does not establish that every clip contains red palm weevil (RPW), or that filtering separates an insect from every other sound. Keep that distinction in filenames, results and presentations.

**User's current assumption:** treat the insect activity as RPW. The `rpw_bursts/` outputs use that assumption and contain detected, cropped, noise-suppressed transients. They are not guaranteed to exclude handling impulses or residual background. Prefer the short-burst package over the previous full-length clips for this bench exercise.

The 100–2,500 Hz filter follows this project's intended sensing band. It is an engineering choice, not a species identifier. Noise attenuation can suppress real impulses or leave unrelated transients. Compare original, bandpassed and denoised audio before choosing a stimulus. Two thousand clips also do not establish two thousand independent insects, logs or recording sessions.

## 1. Identified hardware and remaining checks

The supplied reimbursement PDF identifies the amplifier, supply and cable. The subsequent user-supplied product screenshot identifies the exciter; Dayton's product page confirms its nominal specifications [6]. These are identification/specification records, not measurements of the assembled rig.

| Item | Receipt evidence | What still needs checking |
|---|---|---|
| Amplifier | Bewinner TPA3116D2, advertised as a 100 W bass/subwoofer board; page 7. Item 10's link on page 2 points to Amazon ASIN `B09ZQ2NX9V`. User's subsequent gallery diagram shows one `OUT` pair, `DC IN`, two alternative AUX inputs, bass-frequency and volume controls. | Confirm the physical board matches the diagram, allowed load, supply requirements and passband. The chip name does not identify the complete board circuit. |
| Supply | 12 V DC, 3 A, advertised 5.5 × 2.5 mm plug; page 6. | Label polarity, actual plug fit and compatibility with this board. |
| Audio cable | VENTION 3.5 mm male to 2.5 mm male, 0.5 m; pages 7 and 11. | Actual amplifier input connector and pinout. A 2.5 mm end will not fit a 3.5 mm input. |
| Exciter | Dayton Audio **DAEX25FHE-4**, 25 mm voice coil, **4 Ω nominal**, **24 W RMS power handling**; user-supplied screenshot and Dayton product page [6]. | Actual mounted response and suitable drive level on the test log. The 24 W rating is not a playback target. |

**Check the amplifier passband before connecting the Dayton exciter.** The receipt calls this a bass/subwoofer module. A board-level crossover could remove much of the required 100–2,500 Hz content even though the TPA3116D2 chip can amplify audio. No exact crossover cutoff or bypass is established here. Use a documented full-range output/mode only if this actual board has one. If its only output is low-pass filtered, it cannot reproduce the intended band faithfully; establish a compatible full-range driver path before the replay test. Do not bypass components based on another TPA3116D2 board's schematic.

**What the new gallery diagram establishes:** this pictured module has a single mono speaker output and a control labeled “Bass frequency adjustment,” consistent with the receipt's subwoofer description. It does not show a full-range output or bypass. This is a substantive compatibility concern for faithful replay, not an impedance conclusion. No numeric cutoff is visible. Turning the bass knob to maximum does not establish that 2,500 Hz passes. Without a documented response or a loaded electrical response measurement, use this module only for exploratory vibration tests and do not claim faithful reproduction of the recordings. A full-range amplifier compatible with the exciter's 4 Ω load is the straightforward path if the bass filter cannot be disabled through a documented setting.

TI specifies a 4.5–26 V operating range for the **chip** [4], so 12 V lies within that chip range; this does not certify the Bewinner module or its input jack. The purchased supply's nominal input budget is **12 × 3 = 36 W**, before amplifier losses. The listing's “100 W” is not the output available from this supply and is not a safe exciter setting. Do not increase the supply voltage to chase that advertised rating.

Before power-up, confirm that the actual amplifier output supports **a 4 Ω load** and the purchased supply. Connect the DAEX25FHE-4 as one passive load across the `OUT + / OUT −` pair shown in the diagram. Nominal impedance alone does not establish the mounted system's frequency response or a safe drive level; begin at minimum gain. Verify the physical board's markings rather than relying on diagram orientation alone.

The purchased MCP602E is the sensing front-end op amp. Do not use it or an ESP32 GPIO as the power driver for a speaker exciter.

## 2. Assemble the replay path

### Connections for the supplied gallery diagram

With power disconnected, use the labels below. Left/right positions refer **only to the upright line drawing**, not the rotated photograph above it.

| Connection | Diagram labels and orientation |
|---|---|
| Supply positive | `DC IN +`, right screw of the DC pair. Use the purchased 12 V supply only after confirming board rating and supply polarity. |
| Supply negative | `DC IN −`, left screw of the DC pair. |
| Exciter positive | `OUT +`, left screw of the speaker-output pair, to the exciter's marked positive terminal. |
| Exciter negative | `OUT −`, right screw of the speaker-output pair, to the other exciter terminal. **Do not connect this to DC negative/GND.** |
| Audio source | Use the `AUX IN 1` socket with the correct plug size, **or** the `AUX IN 2` header labeled `L-in / G / R-in`. Do not use both inputs at once or directly short source left/right outputs together. |
| Volume | The right-hand knob in the drawing. Start at minimum. |
| Bass frequency | The left-hand knob. It adjusts the bass filter; it is not evidence of a full-range mode. |
| `PSW` header | Function not adequately specified by this image. Do not bridge it or assume it bypasses filtering. |

The AUX header's `G` is signal ground; it is not the speaker-output return. Keep speaker wiring away from the piezo acquisition input. The stock 3.5-to-2.5 mm cable is usable only if those sizes match the actual source and input sockets.

### Bench sequence

1. With amplifier power off and gain at minimum, connect a computer/phone audio output or USB audio interface to the amplifier's specified audio input. A USB DAC is needed if the computer has no analog audio output.
2. After the passband check in step 1, connect the passive exciter to **one documented compatible amplifier output pair**, exactly as its manual specifies. Do not join stereo outputs or assume a speaker minus terminal is ground. The TPA3116D2 supports bridge-tied output arrangements [4]; an earthed oscilloscope ground clip must not touch a driven speaker terminal. TI documents the grounding principle explicitly for its BTL evaluation board [1]; confirm this board's arrangement and use an appropriate differential measurement method.
3. Use the purchased **12 V/3 A supply only after confirming the module accepts it**, the connector fits and polarity matches the board markings. Confirm that the exciter impedance is a supported load. Never infer safe amplifier voltage or exciter power from the WAV file's digital level.
4. Mount the DAEX25FHE-4 using its **pre-applied 3M VHB contact ring** [6]. Prepare a small flat, dry, dust-free contact area on the test log; do not rely on adhesive against loose bark or a curved surface that prevents full ring contact. Dayton provides four mounting tabs for extra support. Support the frame without restraining the moving assembly, leave vents clear and strain-relieve the two terminal wires. Do not use the rear magnet label as the coupling face. Keep cables off the sensing piezo. Any added mounting plate changes the transfer path and must be recorded.
5. Attach the project's WANGCL piezo/sensor and MCP602E acquisition front end to a fixed marked location on the same log. Record the exciter-to-sensor distance, mounting method, log material and dimensions. If a second piezo and acquisition channel are available, place it near the exciter as a reference and keep that placement fixed.

Signal path:

```text
WAV player → audio interface → power amplifier → exciter → test log
                                                        ↓
                           sensing piezo → MCP602E → ESP32 ADC
```

Keep the exciter power circuit separate from the piezo/ADC signal input. Speaker output must never be connected to the ADC. Preserve the actual front-end bias and input protection; confirm ADC range against the installed ESP32 circuit.

## 3. Choose the playback file

1. Start with one candidate clip from the small playback set, then audition its original and filtered versions. Listen for clear short activity bursts; reject obviously dominated handling, hum, rubbing or speech. Listening alone does not confirm RPW identity.
2. Play WAV with the correct declared sample rate. A resampled 48 kHz copy is for player compatibility; it does not restore frequencies above the original 8 kHz recording's 4 kHz Nyquist limit. Do not change the speed or pitch.
3. Disable player EQ, bass boost, loudness normalization, spatial sound and automatic gain. Keep the same device, player and volume throughout a comparison.
4. Digital peak normalization is only a playback convenience. A level such as −12 dBFS is **not** a calibrated force, acceleration, voltage, safe power limit or natural larval amplitude. Use recorded gain factors when comparing variants.

## 4. Establish a baseline and raise gain gradually

1. Record at least 10 seconds with the amplifier off, then 10 seconds with it on but playing digital silence. Use the project's 8 kHz acquisition setting. A difference indicates electrical or mechanical interference that needs resolving before the replay experiment.
2. Begin the candidate file with amplifier gain at minimum, raise it slowly until the sensing channel shows repeatable bursts above the silence baseline, and stop at the lowest usable level. Record device volume and amplifier gain. Repeat the same file three times without moving anything.
3. **Stop** if the exciter rattles, mounting slips, amplifier faults, components become unusually hot, or the sensing waveform clips/flat-tops. Reduce gain or correct mounting. Check acquisition headroom before increasing stimulus amplitude.
4. If the signal is weak, first inspect attachment and cable contact, then reduce distance. Increasing power is not the first correction. If the signal is distorted, lower gain and check amplifier clipping, sensor/front-end saturation and mechanical rattling separately.
5. Check for an electrical shortcut: keep amplifier on and playing, but mechanically decouple the exciter from the log without changing cables. A substantial residual recorded signal can be airborne pickup or electrical coupling, so it is not evidence of propagation through wood. Keep this control at the same drive setting.

## 5. Compare stimuli fairly

Run amplifier-off background, amplifier-on silence, clean-lab control, original infested clip, bandpassed clip and noise-attenuated clip in a shuffled order. If the small playback directory lacks a clean-lab control, select one documented clean reference file from the package or source manifest. Match the comparison rule in advance: either the same source-referenced digital gain, or an explicitly measured equal reference-piezo level. Do not independently normalize each variant and call the difference natural amplitude.

For an initial bench check use five candidate clips, three repeats each, one mounting and one distance. After that, change **one** variable at a time: distance, drive setting, mounting or background noise. This small subset checks the replay chain before running all 2,000 files.

Inspect the **raw, unfiltered ADC capture** for clipping against the actual ADC limits first. Remove the front-end DC bias (or subtract each capture's mean), then apply the project's intended 100–2,500 Hz acquisition filter before calculating band-limited RMS. For a fixed capture window let `P_on = mean(x_on**2)` and `P_silence = mean(x_silence**2)`, using consistent units and gain. If `P_on > P_silence > 0`, an exploratory replay-excess ratio is:

```text
10 log10((P_on − P_silence) / P_silence)
```

This assumes additive, comparable background noise. It is not ground-truth insect SNR; mark it undefined when the inequality fails. The project's 4 dB SNR gate is an intended processing setting, not a proven RPW threshold. Record the implementation used rather than substituting this estimate silently.

## 6. Save evidence and interpret the result

For every capture save:

```text
trial_id, source_clip_id, variant, source_hash, playback_gain,
player_volume, amplifier_gain, amplifier_supply_v, exciter_model,
exciter_impedance_ohm, mounting, distance_cm, log_id, sensor_id,
sample_rate_hz, front_end_gain, clipping, background_file,
capture_file, reference_capture_file, remarks
```

Append measured reference-piezo level, replay-excess ratio and classifier result if available. Keep raw captures alongside processing outputs. Record any missing calibration explicitly.

A repeatable capture confirms that the mechanical replay reaches this sensing chain under those bench conditions. It does not verify live-RPW detection accuracy, species identity, natural vibration amplitude or field range. Exciter and wood response color the recording. Matching the original waveform quantitatively requires measuring the mounted transfer response and a defined reference level; avoid aggressive inverse equalization around resonant nulls.

## References and project context

1. [Texas Instruments TAS5711EVM guide, §2.1.3](https://www.ti.com/lit/ug/slou280/slou280.pdf): BTL output wiring and grounding principle. This is a reference example, not identification of the purchased amplifier.
2. [TreeVibes dataset](https://www.kaggle.com/datasets/potamitis/treevibes), v1; local lab README and archive manifest provide lab provenance. No species-resolved lab labels were available at preparation.
3. Project `docs/HARDWARE_BASELINE.md`: WANGCL sensing module, MCP602E, ESP-WROOM-32; intended 8 kHz sampling, 100–2,500 Hz passband and 4 dB gate. These settings are project intent, not measured bench performance.
4. [Texas Instruments TPA3116D2 datasheet](https://www.ti.com/lit/ds/symlink/tpa3116d2.pdf): chip-level supply range and BTL/PBTL architecture; not a schematic or passband specification for the Bewinner module.
5. [Supplied reimbursement PDF](</Users/khalidhamdan/Downloads/Reciepts for Senior/Reimbursement Form with Receipts.pdf>), pages 2, 6, 7 and 11: purchased amplifier, supply and cable descriptions. The original PDF has not been changed. [Receipt-linked amplifier listing](https://www.amazon.ae/dp/B09ZQ2NX9V) could not be retrieved for independent specification verification.
6. [Dayton Audio DAEX25FHE-4 product page](https://daytonaudio.vortxhosting.com/product/1177/daex25fhe-4-framed-high-efficiency-25mm-exciter): 4 Ω nominal, 24 W RMS, adhesive-ring mounting and supporting tabs. Dayton states that response depends on the mounting surface; no flat 100–2,500 Hz response on palm wood is established. User's product screenshot identifies the same model.
7. User-supplied amplifier product-gallery screenshot: wiring drawing labeled `DC IN`, `OUT`, `AUX IN 1`, `L-in G R-in`, bass-frequency adjustment and volume. This establishes the depicted terminal arrangement, not measured passband or confirmation of the physical PCB revision.

**Next hardware action:** match the physical board to the supplied diagram, confirm supply polarity and 4 Ω support, and resolve the subwoofer filter. If no documented full-range mode exists, use a compatible full-range amplifier for the five-clip replay check. The exciter and depicted wiring are now identified; faithful playback through this bass board is not established.

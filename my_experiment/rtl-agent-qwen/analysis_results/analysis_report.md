# Evaluation Results Analysis

**Legend:** `✓` = 100% pass, `✗` = 0% fail, `N%` = partial pass

**Consistency:** 🟢 = all passed, 🟡 = ≥50% passed, 🟠 = <50% passed, 🔴 = all failed

## High-Level Summary by Difficulty

| Difficulty | Total | base | ft-easy | ft-hard |
|------------|-------|------|---------|---------|
| **easy** | 17 | 5/14 (36%) | 8/16 (50%) | 9/17 (53%) |
| **medium** | 55 | 3/55 (5%) | 8/55 (15%) | 7/55 (13%) |
| **hard** | 20 | 1/19 (5%) | 0/20 (0%) | 0/20 (0%) |
| **TOTAL** | 92 | 9/88 (10%) | 16/91 (18%) | 16/92 (17%) |

---

## EASY Problems (17 total)

| Problem | base | ft-easy | ft-hard | Solved By |
|---------|------|---------|---------|-----------|
| `arithmetic_progression_generator_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `axis_broadcaster_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `barrel_shifter_0001` | ✓ 🟢 | ✓ ✓ 🟢 | ✓ ✓ ✓ 🟢 | base, ft-easy, ft-hard |
| `barrel_shifter_0002` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `binary_to_gray_0003` | ✓ 🟢 | ✓ ✓ 🟢 | ✓ ✓ ✓ 🟢 | base, ft-easy, ft-hard |
| `caesar_cipher_0001` | ✓ 🟢 | ✓ ✓ 🟢 | ✓ ✓ ✓ 🟢 | base, ft-easy, ft-hard |
| `cellular_automata_0002` | ✗ 🔴 | ✗ ✓ 🟡 | ✗ ✓ ✗ 🟠 | ft-easy, ft-hard |
| `cic_decimator_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `digital_stopwatch_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dual_port_memory_0001` | ✓ 🟢 | ✓ ✓ 🟢 | ✓ ✓ ✓ 🟢 | base, ft-easy, ft-hard |
| `event_scheduler_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✓ ✗ ✗ 🟠 | ft-hard |
| `fixed_arbiter_0010` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `lfsr_0001` | ✓ 🟢 | ✓ ✓ 🟢 | ✓ ✓ ✓ 🟢 | base, ft-easy, ft-hard |
| `lfsr_0005` | ✗ 🔴 | ✓ ✗ 🟡 | ✗ ✓ ✗ 🟠 | ft-easy, ft-hard |
| `nbit_swizzling_0001` | - | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `programmable_fsm_dynamic_state_encoding_0001` | - | ✓ ✓ 🟢 | ✗ ✗ ✓ 🟠 | ft-easy, ft-hard |
| `sorter_0009` | - | - | ✗ ✗ ✗ 🔴 | - |

**EASY Summary:**

| Model | Evaluated | Solved | Solve Rate | Total Evals | Passes | Pass Rate |
|-------|-----------|--------|------------|-------------|--------|-----------|
| base | 14 | 5 | 35.7% | 14 | 5 | 35.7% |
| ft-easy | 16 | 8 | 50.0% | 32 | 14 | 43.8% |
| ft-hard | 17 | 9 | 52.9% | 51 | 19 | 37.3% |

---

## MEDIUM Problems (55 total)

| Problem | base | ft-easy | ft-hard | Solved By |
|---------|------|---------|---------|-----------|
| `64b66b_codec_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✓ 🟠 | ft-hard |
| `AES_encryption_decryption_0003` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `AES_encryption_decryption_0005` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `AES_encryption_decryption_0009` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `DES_0003` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `DES_0005` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `DES_0007` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `Min_Hamming_Distance_Finder_0001` | ✗ 🔴 | ✗ 11% ✗ ✓ 🟠 | ✗ ✗ ✓ 🟠 | ft-easy, ft-hard |
| `PCIe_endpoint_0001` | ✗ 🔴 | ✗ ✓ ✓ ✓ 🟡 | ✓ ✗ ✓ 🟡 | ft-easy, ft-hard |
| `async_fifo_compute_ram_application_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `async_filo_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `axi4lite_to_pcie_config_0003` | ✓ 🟢 | ✗ ✗ ✓ ✗ 🟠 | ✗ ✗ ✗ 🔴 | base, ft-easy |
| `axis_to_uart_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `axis_to_uart_0004` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `bcd_adder_0004` | ✗ 🔴 | ✓ ✗ ✗ ✗ 🟠 | ✗ ✗ ✗ 🔴 | ft-easy |
| `binary_search_tree_algorithms_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `byte_enable_ram_0002` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `cache_controller_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `cipher_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `cont_adder_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `csr_using_apb_interface_0001` | ✗ 🔴 | ✗ ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `custom_fifo_0004` | ✗ 🔴 | ✗ ✓ ✗ ✓ 🟡 | ✗ ✓ ✓ 🟡 | ft-easy, ft-hard |
| `direct_map_cache_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `direct_map_cache_0003` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dma_xfer_engine_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `door_lock_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dual_port_memory_0004` | ✗ 🔴 | ✓ ✗ ✗ 🟠 | ✗ ✓ ✗ 🟠 | ft-easy, ft-hard |
| `elevator_control_0004` | ✗ 🔴 | ✗ ✗ ✓ 🟠 | ✗ ✗ ✗ 🔴 | ft-easy |
| `ethernet_mii_0004` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `event_scheduler_0004` | ✓ 🟢 | ✓ ✓ ✗ 🟡 | ✓ ✗ ✓ 🟡 | base, ft-easy, ft-hard |
| `hdbn_codec_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `low_power_channel_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `multiplexer_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `nmea_gps_0008` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0010` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0013` | 11% 🔴 | ✗ 2% ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0015` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0019` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✓ ✗ 🟠 | ft-hard |
| `phase_rotation_0038` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `poly_decimator_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `prbs_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `queue_0001` | ✗ 🔴 | ✗ ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `rgb_color_space_conversion_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `rgb_color_space_conversion_0004` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `sigma_delta_audio_0001` | ✓ 🟢 | ✗ 33% ✗ 🔴 | ✗ ✗ ✗ 🔴 | base |
| `signed_comparator_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `sorter_0016` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `spi_complex_mult_0002` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `swizzler_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `swizzler_0005` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `sync_serial_communication_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `systolic_array_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `ttc_lite_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `universal_shift_reg_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `universal_shift_reg_0003` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |

**MEDIUM Summary:**

| Model | Evaluated | Solved | Solve Rate | Total Evals | Passes | Pass Rate |
|-------|-----------|--------|------------|-------------|--------|-----------|
| base | 55 | 3 | 5.5% | 55 | 3 | 5.5% |
| ft-easy | 55 | 8 | 14.5% | 172 | 12 | 7.0% |
| ft-hard | 55 | 7 | 12.7% | 165 | 10 | 6.1% |

---

## HARD Problems (20 total)

| Problem | base | ft-easy | ft-hard | Solved By |
|---------|------|---------|---------|-----------|
| `AES_encryption_decryption_0012` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `AES_encryption_decryption_0018` | - | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `DES_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `async_fifo_compute_ram_application_0006` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `binary_search_tree_algorithms_0014` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dynamic_equalizer_0001` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dynamic_equalizer_0004` | ✗ 🔴 | ✗ ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `dynamic_equalizer_0008` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `ethernet_mii_0006` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `event_storing_0001` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `gcd_0007` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `jpeg_runlength_enc_0001` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `monte_carlo_0006` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0028` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `phase_rotation_0031` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `rc5_0001` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `secure_apb_history_shift_register_0001` | ✓ 🟢 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | base |
| `sorter_0026` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `thermostat_secure_0001` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |
| `traffic_light_controller_0001` | ✗ 🔴 | ✗ 🔴 | ✗ ✗ ✗ 🔴 | - |

**HARD Summary:**

| Model | Evaluated | Solved | Solve Rate | Total Evals | Passes | Pass Rate |
|-------|-----------|--------|------------|-------------|--------|-----------|
| base | 19 | 1 | 5.3% | 19 | 1 | 5.3% |
| ft-easy | 20 | 0 | 0.0% | 27 | 0 | 0.0% |
| ft-hard | 20 | 0 | 0.0% | 60 | 0 | 0.0% |

---

## Unique Solves (Problems only one model solved)

### EASY

**ft-hard** (1 unique):
- `event_scheduler_0001`

**Solved by multiple models:**
- `barrel_shifter_0001` (base, ft-easy, ft-hard)
- `binary_to_gray_0003` (base, ft-easy, ft-hard)
- `caesar_cipher_0001` (base, ft-easy, ft-hard)
- `cellular_automata_0002` (ft-easy, ft-hard)
- `dual_port_memory_0001` (base, ft-easy, ft-hard)
- `lfsr_0001` (base, ft-easy, ft-hard)
- `lfsr_0005` (ft-easy, ft-hard)
- `programmable_fsm_dynamic_state_encoding_0001` (ft-easy, ft-hard)

### MEDIUM

**base** (1 unique):
- `sigma_delta_audio_0001`

**ft-easy** (2 unique):
- `bcd_adder_0004`
- `elevator_control_0004`

**ft-hard** (2 unique):
- `64b66b_codec_0001`
- `phase_rotation_0019`

**Solved by multiple models:**
- `Min_Hamming_Distance_Finder_0001` (ft-easy, ft-hard)
- `PCIe_endpoint_0001` (ft-easy, ft-hard)
- `axi4lite_to_pcie_config_0003` (base, ft-easy)
- `custom_fifo_0004` (ft-easy, ft-hard)
- `dual_port_memory_0004` (ft-easy, ft-hard)
- `event_scheduler_0004` (base, ft-easy, ft-hard)

### HARD

**base** (1 unique):
- `secure_apb_history_shift_register_0001`

---

## Overall Model Comparison

| Model | Easy Solved | Medium Solved | Hard Solved | Total Solved |
|-------|-------------|---------------|-------------|--------------|
| base | 5/14 (36%) | 3/55 (5%) | 1/19 (5%) | 9 |
| ft-easy | 8/16 (50%) | 8/55 (15%) | 0/20 (0%) | 16 |
| ft-hard | 9/17 (53%) | 7/55 (13%) | 0/20 (0%) | 16 |
# CSE457 — Báo cáo Lab 1

**Phân tích và xử lý tín hiệu âm thanh số với datalab1.mp3**

Notebook: [CSE457_Lab1_VSCode_datalab1.ipynb](CSE457_Lab1_VSCode_datalab1.ipynb). Dữ liệu: [datalab1.mp3](datalab1.mp3).

Báo cáo này được tạo từ lần chạy notebook, đối chiếu phần A–G của *Lab 1.pdf*, trang 4–5; thí nghiệm độc lập và câu hỏi ở trang 9–10. Dữ liệu do người học cung cấp khác case study, vì vậy kết quả không phải tái tạo các con số của Western Cowboy Texas Music.mp3.

## Môi trường và quy ước

- Python 3.10.5; NumPy 2.2.6; SciPy 1.15.3; SoundFile 0.14.0; Matplotlib 3.10.9; pandas 2.3.3.
- Đặt notebook và MP3 cùng thư mục, cài `python -m pip install -r requirements.txt`, chọn kernel rồi **Run All**.
- Mẫu tính toán float64. Biên độ chuẩn hóa theo full scale, RMS dBFS = 20log10(RMS/1); phổ một phía hiệu chỉnh tổng cửa sổ và ghi dB re biên độ đỉnh 1 FS. Spectrogram dùng dB re S=1 theo scaling spectrum; đáp ứng cửa sổ ghi relative dB, SNR ghi SNR dB.
- Không tự chuẩn hóa riêng từng tín hiệu khi so sánh mức. Trình phát notebook dùng `normalize=False` trên cùng đoạn; giữ cùng âm lượng thiết bị khi nghe.
- **Giới hạn nghe thử:** đã cung cấp audio và trình phát để so sánh; các mô tả cảm nhận là dự đoán dựa trên phổ/đáp ứng/nhiễu, chưa phải kết quả nghe chủ quan hoặc khảo sát người nghe. Không có bản thu lossless trước MP3 để đo tổn thất mã hóa.


## A. Metadata và stereo–mono

Tệp có 2 kênh, Fs = 44100 Hz và thời lượng 214.343401 s; miền biểu diễn độc lập kết thúc ở 22050 Hz. RMS các kênh là 0.160368, 0.159937, còn mono là 0.143072 (-16.89 dBFS), cho thấy việc lấy trung bình kênh làm thay đổi mức tín hiệu. Gain chuẩn hóa chung bằng 1; không chuẩn hóa riêng từng kênh nên vẫn giữ tương quan mức giữa chúng. MP3 không có bit depth PCM cố định; float64 là kiểu tính toán, còn PCM16 là cấu hình xuất tham chiếu.

| Thuộc tính | Giá trị | Đơn vị |
| --- | --- | --- |
| File | datalab1.mp3 |  |
| Format | MP3 |  |
| Subtype | MPEG_LAYER_III |  |
| Sampling rate | 44100 | Hz |
| Channels | 2 |  |
| Duration | 214.343 | s |
| Decoded dtype | float64 |  |
| MP3 size | 3697728 | byte |
| PCM reference | 16 | bit/sample |
| Common normalization gain | 1 |  |

| Tín hiệu | Peak | RMS | RMS_dBFS |
| --- | --- | --- | --- |
| Kênh 1 | 0.937987 | 0.160368 | -15.8976 |
| Kênh 2 | 0.923114 | 0.159937 | -15.921 |
| Mono | 0.85309 | 0.143072 | -16.8889 |

![Các kênh và mono: cùng thời gian, gain và thang biên độ.](Lab01_output/figures/A_stereo_mono.png)

*Các kênh và mono: cùng thời gian, gain và thang biên độ.*

## B. Phân tích miền thời gian

Peak mono = 0.853090, RMS = 0.143072 và năng lượng ∑x² = 193489.828; có 0 mẫu mono đạt ngưỡng |x| ≥ 0,999 nên kiểm tra này không phát hiện nguy cơ clipping tại full scale. Đoạn 0–1 s có RMS = 0.000000, trong khi đoạn 120–121 s có RMS = 0.204232; hình cho thấy khác biệt rõ về mức năng lượng. Với đoạn im lặng RMS = 0, mức RMS là −∞ dBFS, không diễn giải sàn số dùng khi vẽ log là âm thanh thật. Các số liệu được tính từ toàn bộ mẫu; chỉ waveform toàn tệp giảm số điểm khi hiển thị.

| Peak_mono | RMS_mono | RMS_dBFS | Energy_sum_x2 | Mono_samples_abs_ge_0.999 | Input_channel_samples_abs_ge_1 |
| --- | --- | --- | --- | --- | --- |
| 0.85309 | 0.143072 | -16.8889 | 193490 | 0 | 0 |

| start_s | end_s | RMS | Peak | RMS_dBFS |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | -inf |
| 120 | 121 | 0.204232 | 0.640338 | -13.7975 |

![Waveform toàn tệp, zoom 1 giây và hai đoạn tương phản.](Lab01_output/figures/B_waveforms.png)

*Waveform toàn tệp, zoom 1 giây và hai đoạn tương phản.*

Audio: [segment_low_rms.wav](Lab01_output/audio/segment_low_rms.wav) · [segment_high_rms.wav](Lab01_output/audio/segment_high_rms.wav)

## C. Phân tích FFT

Đoạn 140.250–141.000 s được chọn bằng điểm ổn định nhỏ nhất trong các đoạn hoạt động: CV của RMS = 0.0912, biến đổi phổ trung bình = 0.3519; đây là thước đo ổn định tương đối. Hai NFFT [65536, 131072] dùng nguyên 33075 mẫu và cùng cửa sổ Hamming, không cắt mẫu; Δf giảm từ 0.672913 xuống 0.336456 Hz. Các đỉnh nổi bật ở 58.21, 116.08, 328.72, 440.08, 586.11 Hz, nhưng không đủ cơ sở quy tất cả cho một nguồn âm. Zero-padding chỉ làm lưới phổ dày hơn; khả năng tách hai tần số vẫn phụ thuộc độ dài 0.750 s và main-lobe của Hamming.

| start_s | duration_s | samples_used | NFFT | bin_spacing_Hz |
| --- | --- | --- | --- | --- |
| 140.25 | 0.75 | 33075 | 65536 | 0.672913 |
| 140.25 | 0.75 | 33075 | 131072 | 0.336456 |

| frequency_Hz | peak_amplitude_FS | dB_re_peak_FS1 |
| --- | --- | --- |
| 58.2069 | 0.133984 | -17.4589 |
| 116.077 | 0.0282778 | -30.9711 |
| 328.718 | 0.0328816 | -29.6609 |
| 440.085 | 0.0271822 | -31.3143 |
| 586.107 | 0.0320403 | -29.8861 |

![FFT tuyến tính và dB; hiệu chỉnh gain cửa sổ, dùng cùng chuẩn 1 FS.](Lab01_output/figures/C_fft_linear_db.png)

*FFT tuyến tính và dB; hiệu chỉnh gain cửa sổ, dùng cùng chuẩn 1 FS.*

## D. STFT và thay đổi thời gian–tần số

Ba spectrogram dùng cùng tín hiệu 5–15 s, NFFT=4096, hop 10 ms và chung khoảng màu [-90.31, -10.31] dB re S=1. Với frame 25 ms, khoảng 14.452–14.952 s có biến đổi phổ trung bình nhỏ nhất trên cửa sổ khoảng 0,5 s; tại 5.512 s có biến đổi phổ giữa hai frame lớn nhất, là ứng viên vùng chuyển tiếp nhanh để quan sát. Trong biểu diễn 25 ms, khoảng 96.17% tổng S² nằm dưới 2 kHz, giải thích các vùng màu mạnh ở dải thấp. Frame 10 ms dùng ít dữ liệu nên dải tần rộng hơn nhưng ít trộn các sự kiện gần nhau theo thời gian; frame 50 ms làm dải tần hẹp hơn và kéo dài ảnh hưởng của chuyển tiếp, còn 25 ms là mức trung gian.

| frame_ms_requested | frame_samples | frame_ms_actual | hop_samples | overlap_percent | NFFT |
| --- | --- | --- | --- | --- | --- |
| 10 | 441 | 10 | 441 | 0 | 4096 |
| 25 | 1102 | 24.9887 | 441 | 59.9819 | 4096 |
| 50 | 2205 | 50 | 441 | 80 | 4096 |

![Giữ tín hiệu, hop, NFFT, cửa sổ và thang màu. Vạch trắng nét đứt: biến đổi phổ lớn nhất; khung trắng: khoảng ổn định tương đối.](Lab01_output/figures/D_spectrogram_frames.png)

*Giữ tín hiệu, hop, NFFT, cửa sổ và thang màu. Vạch trắng nét đứt: biến đổi phổ lớn nhất; khung trắng: khoảng ổn định tương đối.*

## E. Thí nghiệm cửa sổ

Trên cùng frame 140.450 s, cả hai phổ dùng 1102 mẫu, NFFT=8192 và hiệu chỉnh theo tổng cửa sổ để so sánh biên độ. Đo trên đáp ứng cửa sổ cho thấy main-lobe Rectangular rộng khoảng 80.08 Hz, còn Hamming khoảng 160.15 Hz theo khoảng cách giữa hai cực tiểu đầu tiên. Side-lobe lớn nhất tương ứng -13.26 và -42.67 dB tương đối: Hamming giảm rò phổ nhưng làm các thành phần gần nhau dễ chồng lấn main-lobe. Đáp ứng cửa sổ bổ sung bằng chứng định lượng; không quy mọi gợn trên phổ âm thanh thực cho leakage.

| window | samples | main_lobe_null_to_null_Hz | max_side_lobe_relative_dB |
| --- | --- | --- | --- |
| Rectangular | 1102 | 80.0766 | -13.2615 |
| Hamming | 1102 | 160.153 | -42.6745 |

![Phổ frame thực và đáp ứng hai cửa sổ; cùng tham số trong mỗi phép so sánh.](Lab01_output/figures/E_windows.png)

*Phổ frame thực và đáp ứng hai cửa sổ; cùng tham số trong mỗi phép so sánh.*

## F. Lọc số FIR

Hai bộ lọc đều dài 201 taps, bậc 200, có group delay 100 mẫu = 2.267574 ms. Gain lớn nhất trong stopband đo được là -57.40 dB cho LPF ở ≥2,5 kHz và -20.04 dB cho HPF ở 0–100 Hz; không coi cutoff là vách cắt lý tưởng. Phổ LPF giảm dải cao và HPF giảm dải thấp phù hợp với H(f); các đường dùng chung chuẩn 1 FS và đầu ra được bù trễ khi so sánh, còn WAV lưu đầu ra nhân quả. Dự đoán khi nghe: LPF làm âm kém sáng do mất thành phần cao, HPF giảm tiếng trầm/ù và có thể làm giọng mỏng hơn; đây chưa phải kết quả khảo sát người nghe.

| filter | taps | order | group_delay_ms | stopband_used_Hz | worst_stopband_gain_dB |
| --- | --- | --- | --- | --- | --- |
| LPF 2 kHz | 201 | 200 | 2.26757 | >=2500 | -57.4018 |
| HPF 300 Hz | 201 | 200 | 2.26757 | 0..100 | -20.0351 |

![Đáp ứng bộ lọc và phổ cùng chuẩn biên độ, cùng sự kiện sau bù trễ.](Lab01_output/figures/F_filter_response_spectrum.png)

*Đáp ứng bộ lọc và phổ cùng chuẩn biên độ, cùng sự kiện sau bù trễ.*

Audio: [input_mono_PCM16.wav](Lab01_output/audio/input_mono_PCM16.wav) · [filtered_lpf_2k.wav](Lab01_output/audio/filtered_lpf_2k.wav) · [filtered_hpf_300Hz.wav](Lab01_output/audio/filtered_hpf_300Hz.wav)

## G1. Lượng tử hóa và SNR

Thí nghiệm dùng bộ lượng tử PCM với Δ = 2/2^B và 2^B mức từ −1 đến 1−Δ; khác công thức qmax đối xứng trong ví dụ đề bài, lựa chọn này khớp chính xác lưới của WAV PCM. SNR đo được ở 4/8/16 bit lần lượt 12.177, 36.173, 84.295 dB, tăng 23.996 dB khi thêm 4 bit và 48.123 dB khi thêm 8 bit, gần xu hướng 6 dB/bit. Đọc lại WAV cho sai khác bằng 0 so với mảng lượng tử, nên SNR trong bảng thực sự đại diện file xuất; bản 4-bit lưu trong PCM16 chỉ mô phỏng mức lượng tử, không tiết kiệm dung lượng như dữ liệu đóng gói 4-bit. Dự đoán nhiễu/méo lượng tử dễ nhận ra ở phần âm nhỏ hoặc đuôi âm do tỷ lệ sai số/tín hiệu lớn; im lặng số đúng bằng 0 vẫn lượng tử thành 0 khi không dither.

| B_bit | step_FS | levels_available | SNR_array_dB | SNR_WAV_dB | noise_RMS | SNR_model_dB | max_WAV_array_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.125 | 16 | 12.1767 | 12.1767 | 0.0352143 | 11.9647 | 0 |
| 8 | 0.0078125 | 256 | 36.1728 | 36.1728 | 0.00222287 | 36.0471 | 0 |
| 16 | 3.05176e-05 | 65536 | 84.2955 | 84.2955 | 8.72531e-06 | 84.2119 | 0 |

![SNR đo trực tiếp từ WAV và dạng sóng lượng tử; cùng full scale.](Lab01_output/figures/G1_quantization.png)

*SNR đo trực tiếp từ WAV và dạng sóng lượng tử; cùng full scale.*

Audio: [quantized_4bit_levels_in_PCM16.wav](Lab01_output/audio/quantized_4bit_levels_in_PCM16.wav) · [quantized_8bit.wav](Lab01_output/audio/quantized_8bit.wav) · [quantized_16bit.wav](Lab01_output/audio/quantized_16bit.wav)

## G2. Resampling 16 kHz và 8 kHz

Sau resample, Nyquist giảm còn 8 kHz và 4 kHz; thời lượng chỉ khác ở mức làm tròn một mẫu, còn phổ dùng cùng chuẩn biên độ và cùng khoảng thời gian. Trong thí nghiệm sinusoid kiểm soát, tone 12 kHz sẽ gập về 4 kHz khi lấy mẫu trực tiếp ở 16 kHz, và tone 6 kHz gập về 2 kHz khi lấy mẫu trực tiếp ở 8 kHz; resample_poly giảm biên độ alias tương ứng 69.70 và 69.70 dB sau khi bỏ vùng biên. Kết quả cho thấy lọc chống aliasing có tác dụng, nhưng FIR hữu hạn không đảm bảo alias bằng 0 tuyệt đối ở mọi tần số và thí nghiệm tone không đo toàn bộ alias của bản ghi thực. Dự đoán bản 8 kHz kém sáng/rõ chi tiết cao hơn bản 16 kHz, đặc biệt ở thành phần trên 4 kHz; mức giảm chất lượng cảm nhận cần xác nhận bằng các trình phát cùng âm lượng trong notebook.

| Fs_Hz | samples | duration_s | Nyquist_Hz | Peak |
| --- | --- | --- | --- | --- |
| 16000 | 3429495 | 214.343 | 8000 | 0.841624 |
| 8000 | 1714748 | 214.344 | 4000 | 0.770467 |

| target_Fs_Hz | input_tone_Hz | alias_Hz | alias_amp_without_filter | alias_amp_resample_poly | alias_change_dB |
| --- | --- | --- | --- | --- | --- |
| 16000 | 12000 | 4000 | 1 | 0.000327236 | -69.7028 |
| 8000 | 6000 | 2000 | 1 | 0.000327236 | -69.7028 |

![Phổ bản gốc và hai sampling rate.](Lab01_output/figures/G2_resampling.png)

*Phổ bản gốc và hai sampling rate.*

![Thí nghiệm phụ có kiểm soát, cùng sinusoid và cùng mức biên độ trước khi đổi Fs.](Lab01_output/figures/G2_alias_control.png)

*Thí nghiệm phụ có kiểm soát, cùng sinusoid và cùng mức biên độ trước khi đổi Fs.*

Audio: [speech_resampled_16kHz.wav](Lab01_output/audio/speech_resampled_16kHz.wav) · [speech_resampled_8kHz.wav](Lab01_output/audio/speech_resampled_8kHz.wav)

## G3. Bit rate, kích thước và mức nén

PCM16 2 kênh tại 44100 Hz có R = Fs × B × C = 1,411,200 bit/s, payload 37,810,176 byte cho 214.343401 s; WAV thực tế lớn hơn payload 44 byte do phần chứa/header. MP3 có kích thước 3,697,728 byte và bitrate trung bình ước lượng 138.011 kbps, tương ứng tỷ lệ PCM/MP3 = 10.225245:1 và tiết kiệm 90.2203%. Ước lượng bitrate từ toàn tệp bao gồm metadata; PCM và MP3 được so sánh với cùng số kênh, Fs và thời lượng, không so stereo với mono. Giải mã MP3 sang WAV chỉ đổi biểu diễn lưu trữ; do không có bản thu lossless trước mã hóa, không thể đo tổn thất codec hoặc kết luận WAV khôi phục/chất lượng cao hơn MP3 từ phép so sánh này.

| Đại lượng | Giá trị | Đơn vị |
| --- | --- | --- |
| PCM bitrate | 1.4112e+06 | bit/s |
| MP3 average file bitrate | 138011 | bit/s |
| PCM theoretical payload | 3.78102e+07 | byte |
| WAV actual size | 3.78102e+07 | byte |
| MP3 actual size | 3.69773e+06 | byte |
| PCM/MP3 ratio | 10.2252 | :1 |
| Saving | 90.2203 | % |

![Bitrate và kích thước: MB theo hệ thập phân; đối chiếu cùng cấu hình.](Lab01_output/figures/G3_coding.png)

*Bitrate và kích thước: MB theo hệ thập phân; đối chiếu cùng cấu hình.*

Audio: [input_decoded_PCM16.wav](Lab01_output/audio/input_decoded_PCM16.wav)

## Trả lời 7 câu hỏi báo cáo

1. **Vì sao Fs = 44,1 kHz biểu diễn độc lập tới 22,05 kHz?** Sau lấy mẫu, exp(j2π(f+kFs)n/Fs) = exp(j2πfn/Fs), nên các tần số cách nhau bội Fs có cùng mẫu. Với tín hiệu thực, phần tần số âm là liên hợp của phần dương; dải độc lập là 0 đến Fs/2 = 22.050 Hz. Thành phần trên Nyquist gập vào dải này nếu không được lọc trước; tại đúng Nyquist còn có trường hợp pha không thể khôi phục đầy đủ, nên thực tế cần băng chuyển tiếp.

2. **NFFT từ 2048 lên 8192 với frame 25 ms:** Δf giảm từ 44100/2048 = 21.533203 Hz xuống 5.383301 Hz, tức lưới phổ dày hơn 4 lần. Frame vẫn khoảng 1102 mẫu (~25 ms), nên lượng thông tin và main-lobe do cửa sổ quyết định không thay đổi. Không thể dùng zero-padding để tách hai thành phần vốn bị chồng main-lobe.

3. **Hamming và leakage:** phép cắt đoạn tương đương nhân cửa sổ, nên phổ quan sát là phổ tín hiệu chập với phổ cửa sổ. Hamming làm nhỏ biên ở hai đầu frame và hạ side-lobe, giảm rò năng lượng sang các tần số khác. Đổi lại main-lobe rộng hơn Rectangular, làm hai đỉnh gần nhau khó phân tách; số đo ở phần E minh họa hai mặt của đánh đổi này.

4. **Độ trễ FIR:** (201−1)/2 = 100 mẫu, tương đương 100/44100 × 1000 = 2.267574 ms. Trong xử lý offline có thể dịch để căn chỉnh, nhưng xử lý nhân quả thời gian thực phải chờ mẫu; tổng latency còn có buffering, frame, thiết bị và các khối nối tiếp. Khoảng 2,27 ms là nhỏ trong nhiều ứng dụng, nhưng vẫn đáng tính vào ngân sách trễ khi monitoring, tương tác hoặc ghép với tín hiệu trực tiếp.

5. **Ảnh hưởng của B và σx:** SNR_Q ≈ 6B + 4,77 − 20log10(Xmax/σx). Giữ Xmax và tín hiệu không đổi, thêm một bit tăng SNR khoảng 6 dB; nếu giảm RMS σx một nửa trong khi bước lượng tử không đổi, SNR giảm khoảng 6,02 dB. Với tín hiệu thực có im lặng, phân bố không đều, clipping hoặc tương quan sai số, mô hình nhiễu đều chỉ gần đúng; dùng SNR đo ở G1 để đánh giá dữ liệu thật.

6. **PCM 60 s và MP3 128 kbps:** PCM = 44100 × 16 × 2 × 60 / 8 = 10.584.000 byte = 10,584 MB ≈ 10.094 MiB. MP3 128.000 bit/s dài 60 s có payload xấp xỉ 960.000 byte = 0,960 MB, chưa tính header/metadata. Tỷ lệ = 11,025:1, tiết kiệm khoảng 90.9297%; không nhầm MB thập phân với MiB.

7. **Hai trường hợp nghe tốt hơn không tương đương SNR cao hơn:** (a) Sai số được dồn vào vùng bị âm mạnh che lấp có thể ít nghe thấy hơn sai số nhỏ hơn nhưng nằm ở vùng tai nhạy; SNR năng lượng toàn dải không tính masking. (b) Dither trước lượng tử có thể làm tăng năng lượng nhiễu, giảm SNR nhưng làm méo lượng tử bớt có tính chu kỳ, giảm các vệt/âm giả khó chịu. Các ví dụ này giải thích giới hạn của SNR, không phải kết quả thử nghe đã thực hiện trên tệp hiện tại.


## Kiểm tra khả năng tái lập và tệp kết quả

Các kiểm tra tự động đều đạt: FFT không cắt mẫu, ít nhất 3 đỉnh, WAV lượng tử giữ chính xác mảng tính toán, thời lượng resample sai khác không quá một mẫu, FIR đối xứng và tất cả đầu ra được tạo.

Lần chạy này tạo **10 hình** và **11 WAV** trong `Lab01_output/`. Danh sách audio: [audio_manifest.csv](Lab01_output/audio_manifest.csv); tham số và kiểm tra: [verification.json](Lab01_output/verification.json). Các bảng CSV riêng lưu metadata, RMS kênh, FFT, STFT, cửa sổ, FIR, lượng tử, resampling, alias và coding.

SHA-256 đầu vào: `1e4bf89d822f308c38c522dac1986bffc8df701fb14b3b0d538a4cbbd2b9c9cb`.

Đề yêu cầu báo cáo Markdown và nộp công khai trên GitHub (trang 10). Bộ tệp hiện được chuẩn bị cục bộ; cần điền thông tin sinh viên/nhóm và đặt vào thư mục nộp theo quy định của lớp. Báo cáo không khẳng định đã xuất bản lên GitHub.

## Nguồn

- *CSE457 — Lab 1. Phân tích và xử lý tín hiệu âm thanh số*, tài liệu người học cung cấp: công thức trang 2–4; yêu cầu A–G trang 4–5; ví dụ lượng tử trang 8; thí nghiệm, câu hỏi và cấu trúc nộp trang 9–10.
- Tất cả bảng và hình trong báo cáo được tính từ `datalab1.mp3` bằng mã nguồn đi kèm. Phép thử sinusoid ở G2 là tín hiệu tổng hợp nhằm kiểm soát tần số alias, không phải nội dung của MP3.

# Nguồn gốc và quy trình tạo dữ liệu

Tài liệu này mô tả nguồn dữ liệu, các bước xử lý và mối quan hệ giữa bộ dữ liệu trong repository này với quy trình tại [`goldvn-data-pipeline`](https://github.com/htrang22/goldvn-data-pipeline).

## 1. Mục đích

Ba bộ dữ liệu trong repository được chuẩn bị để phân tích lợi suất, biến động và mức độ liên kết giữa:

- Bitcoin quy đổi sang VND;
- vàng miếng SJC tại TPHCM;
- VN-Index.

Các bộ dữ liệu này được sử dụng trong notebook `btc-sjc-vnindex-volatility.ipynb` để thực hiện các phân tích ARMA–EGARCH, DCC–GARCH và Diebold–Yılmaz.

## 2. Quy trình xử lý dữ liệu

Mã nguồn tải, làm sạch và tính lợi suất được quản lý riêng tại:

- Repository: [`goldvn-data-pipeline`](https://github.com/htrang22/goldvn-data-pipeline)
- Phiên bản pipeline tham chiếu: [`1948ea0`](https://github.com/htrang22/goldvn-data-pipeline/commit/1948ea0c8e0020352532787abd8def87c0975212)
- Khoảng dữ liệu mặc định: từ `2023-08-01` đến hết `2026-08-01`.

Repository hiện tại chỉ chứa các dataset đã qua xử lý dùng cho mô hình. Dữ liệu đầu vào và logic xử lý được quản lý tại repository pipeline.

## 3. Nguồn dữ liệu

| Tài sản hoặc biến | Nguồn | Mã hoặc phạm vi |
| --- | --- | --- |
| Bitcoin | Yahoo Finance | `BTC-USD` |
| Tỷ giá USD/VND | Yahoo Finance | `VND=X` |
| VN-Index | KBS thông qua thư viện `vnstock` | `VNINDEX` |
| Vàng | Dữ liệu giá công khai của PNJ, do người dùng tự chuẩn bị dưới dạng CSV | SJC, TPHCM |

Tỷ giá USD/VND được sử dụng để quy đổi giá đóng cửa Bitcoin từ USD sang VND.

## 4. Quy tắc làm sạch

### Bitcoin

- Chuẩn hóa ngày về định dạng `YYYY-MM-DD`.
- Chuyển các cột `Open`, `High`, `Low` và `Close` sang dạng số.
- Loại các dòng thiếu ngày, thiếu giá hoặc có giá không dương.
- Kiểm tra tính nhất quán của các cột OHLC.
- Loại ngày trùng và giữ quan sát cuối cùng của mỗi ngày.
- Giữ lịch giao dịch hằng ngày của Bitcoin.

### Tỷ giá USD/VND

- Chuẩn hóa ngày và giá đóng cửa.
- Loại giá trị thiếu, vô hạn hoặc không dương.
- Ghép tỷ giá vào lịch Bitcoin theo ngày.
- Nếu tỷ giá thiếu vào cuối tuần hoặc ngày nghỉ, sử dụng tỷ giá hợp lệ gần nhất trước đó bằng phương pháp forward-fill.
- Cột `fx_observed` cho biết tỷ giá có được quan sát trực tiếp trong ngày hay không.
- Cột `fx_filled` cho biết tỷ giá đã được điền từ ngày trước hay chưa.

### Vàng SJC

- Chỉ giữ các quan sát có `product = SJC` và `location = TPHCM`.
- Chuẩn hóa ngày từ cột `query_date`.
- Loại các quan sát thiếu giá, có giá không dương hoặc giá bán thấp hơn giá mua.
- Nếu có nhiều quan sát trong cùng một ngày, giữ bản ghi cập nhật sau cùng.
- Giá đại diện của vàng được tính bằng trung bình giá mua và giá bán:

$$
P_t^{gold} = \frac{buy_t + sell_t}{2}
$$

Đơn vị giá được giữ nguyên theo file nguồn. Việc nhân toàn bộ chuỗi giá với cùng một hệ số không làm thay đổi log-return.

### VN-Index

- Sử dụng giá đóng cửa của chỉ số.
- Chuẩn hóa ngày và chuyển giá về dạng số.
- Loại quan sát thiếu hoặc có giá không dương.
- Loại ngày trùng và giữ quan sát cuối cùng của mỗi ngày.
- Giữ nguyên lịch giao dịch riêng của VN-Index.

## 5. Quy đổi và tính lợi suất

Giá Bitcoin theo VND được tính bằng:

$$
P_t^{BTC,VND} = P_t^{BTC,USD} \times FX_t^{USD/VND}
$$

Lợi suất của từng tài sản được tính dưới dạng log-return phần trăm:

$$
r_t = 100 \times \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Trong đó:

- Bitcoin sử dụng giá Bitcoin đã quy đổi sang VND;
- vàng sử dụng trung bình giá mua và giá bán;
- VN-Index sử dụng giá đóng cửa.

Lợi suất được tính trên lịch quan sát riêng của từng tài sản. Ba chuỗi không được inner-join trước khi tính return. Việc đồng bộ ngày giữa các tài sản, nếu cần, được thực hiện tại giai đoạn chạy mô hình.

Quan sát đầu tiên của mỗi chuỗi không có return vì không tồn tại giá của kỳ trước.

## 6. Danh mục dữ liệu đầu ra

Thông tin dưới đây được kiểm kê ngày `2026-08-14`.

| File | Số quan sát | Khoảng ngày | Biến lợi suất |
| --- | ---: | --- | --- |
| `3y-bitcoin-vnd-returns.csv` | 1.097 | `2023-08-01` – `2026-08-01` | `btc_return` |
| `3y-gold-returns.csv` | 910 | `2023-08-01` – `2026-08-01` | `gold_return` |
| `3y-vnindex-returns.csv` | 748 | `2023-08-01` – `2026-07-31` | `vnindex_return` |

VN-Index kết thúc tại ngày giao dịch gần nhất trước ngày kết thúc cấu hình. Bitcoin có dữ liệu hằng ngày, trong khi vàng và VN-Index chỉ có dữ liệu vào các ngày tương ứng có quan sát.

## 7. Xác nhận phiên bản dữ liệu

Mỗi file được ghi kèm một mã SHA-256, có thể hiểu là dấu vân tay kỹ thuật số của file. Mã này giúp người sử dụng xác nhận rằng dataset họ tải xuống giống chính xác với phiên bản được sử dụng trong nghiên cứu.

Nếu nội dung file thay đổi, kể cả chỉ một giá trị, mã SHA-256 cũng sẽ thay đổi. Khi cập nhật dataset có chủ đích, bảng mã dưới đây cần được cập nhật tương ứng.

| File | SHA-256 |
| --- | --- |
| `3y-bitcoin-vnd-returns.csv` | `0d868a7ca91edc3a2015c18fbc76d992aa42f1cc0d4c6c6ef5565ec73990750c` |
| `3y-gold-returns.csv` | `9a1cd7c701a29932ecff2f61d76f688f941f4487bc07bf3327648c5a873781ce` |
| `3y-vnindex-returns.csv` | `e802bdb304789e01adac74a2a7d50c23704eac211d3e8136ce3e9d80b81128b4` |

## 8. Dữ liệu vàng và quyền sử dụng

Repository `goldvn-data-pipeline` và repository này không lưu trữ hoặc phân phối file CSV chứa dữ liệu lịch sử giá vàng thô được thu thập từ PNJ.

Chủ dự án chưa nhận được giấy phép hoặc văn bản xác nhận từ PNJ cho phép tái phân phối bộ dữ liệu lịch sử giá vàng. File dữ liệu đầu vào phải do người sử dụng tự chuẩn bị và việc thu thập, sử dụng hoặc phân phối dữ liệu phải tuân thủ điều khoản của nguồn và pháp luật áp dụng.

Mã crawler được giữ tại repository pipeline nhằm ghi nhận phương pháp hình thành dataset và phục vụ tham khảo kỹ thuật. Việc công khai mã crawler không cấu thành sự cho phép thu thập hoặc tái phân phối dữ liệu từ PNJ.

Tên PNJ, nhãn hiệu và các quyền liên quan thuộc về chủ sở hữu tương ứng. Dự án không được PNJ tài trợ, chứng thực hoặc liên kết chính thức.

## 9. Hạn chế

- Lịch giao dịch giữa Bitcoin, vàng và VN-Index không đồng bộ.
- Tỷ giá USD/VND thiếu vào cuối tuần hoặc ngày nghỉ được forward-fill.
- Chênh lệch thời điểm ghi nhận giá giữa các thị trường có thể ảnh hưởng đến kết quả phân tích.
- Giá vàng được đại diện bằng trung bình giá mua và giá bán, không nhất thiết tương ứng với giá thực hiện của một giao dịch cụ thể.
- Dữ liệu có thể được nguồn ban đầu điều chỉnh hoặc cập nhật sau thời điểm thu thập.
- Bộ dữ liệu và kết quả phân tích không cấu thành khuyến nghị đầu tư.

## 10. Tái lập kết quả

Để tạo lại dữ liệu:

1. Clone repository [`goldvn-data-pipeline`](https://github.com/htrang22/goldvn-data-pipeline).
2. Cài đặt các thư viện trong `requirements.txt`.
3. Tự chuẩn bị file giá vàng và đặt tại `data/raw/3y-sjc.csv`.
4. Chạy `python3 run_pipeline.py`.
5. Lấy ba file kết quả trong `data/final/`.
6. Đối chiếu tên cột, số quan sát, khoảng ngày và mã SHA-256 với tài liệu này.

Khi chạy lại pipeline với nguồn dữ liệu hoặc thời điểm tải khác, kết quả có thể thay đổi. Vì vậy, mỗi phiên bản dataset nên được ghi kèm commit pipeline, ngày chạy và mã SHA-256 tương ứng.

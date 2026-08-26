# Econometrics: Phân tích rủi ro và liên kết tài chính

Dự án phân tích lợi suất và biến động của ba tài sản tại Việt Nam: **Bitcoin (quy đổi VND), vàng SJC và VN-Index**.

Phân tích được tổ chức theo hai nhánh phương pháp bổ sung cho nhau:

1. [`btc-sjc-vnindex-volatility.ipynb`](./btc-sjc-vnindex-volatility.ipynb) là
   phân tích nền, gồm **ARMA–EGARCH**, **DCC–GARCH** và mô hình
   **VAR hệ số cố định + GFEVD/Diebold–Yılmaz**.
2. [`tvp-var-connectedness.ipynb`](./tvp-var-connectedness.ipynb) là phần mở
   rộng, dùng **TVP–VAR + GFEVD thay đổi theo thời gian** để theo dõi diễn biến
   của TCI và vai trò truyền/nhận rủi ro. Phần này bổ sung cho mô hình VAR tĩnh,
   không thay thế kết quả nền.

Phân tích cũng sử dụng các kiểm định ADF, Phillips–Perron, KPSS, Zivot–Andrews, Ljung–Box và ARCH-LM; đồng thời kiểm tra độ vững theo nhiều kỳ dự báo.

## Cấu trúc dự án

```text
btc-sjc-vnindex-risk-analysis/
├── 3y-bitcoin-vnd-returns.csv
├── 3y-gold-returns.csv
├── 3y-vnindex-returns.csv
├── btc-sjc-vnindex-volatility.ipynb   # Mô hình 1, 2 và VAR-DY tĩnh
├── tvp-var-connectedness.ipynb        # Mở rộng TVP-VAR connectedness
├── outputs/
│   └── model2_conditional_volatility.csv  # Đầu vào chung cho Model 3
├── DATA_PROVENANCE.md
├── requirements.txt
└── README.md
```

## Quy trình phương pháp

```text
Giá tài sản
    ↓
Lợi suất
    ↓
ARMA–EGARCH → conditional volatility của BTC, SJC và VN-Index
    ↓
    ├── DCC–GARCH → tương quan động
    │
    └── Connectedness
        ├── Static VAR → GFEVD → DY (TCI, FROM, TO, NET toàn mẫu)
        └── TVP–VAR   → TVP-GFEVD → TCI_t và NET_t theo thời gian
```

Nhánh static VAR–DY trả lời câu hỏi về cấu trúc lan truyền trung bình trong toàn
mẫu. Nhánh TVP–VAR kiểm tra cấu trúc đó thay đổi khi nào và với cường độ ra sao;
do đó hai đặc tả nên được đọc như baseline và extension/alternative specification.

## Dữ liệu

- `3y-bitcoin-vnd-returns.csv`: giá và lợi suất Bitcoin theo VND.
- `3y-gold-returns.csv`: giá và lợi suất vàng SJC.
- `3y-vnindex-returns.csv`: chỉ số và lợi suất VN-Index.

### Quy trình tạo dữ liệu

Ba file dữ liệu lợi suất trong repository này được tạo bằng quy trình xử lý tại:

- [goldvn-data-pipeline](https://github.com/htrang22/goldvn-data-pipeline)

Pipeline thực hiện tải dữ liệu thị trường, đọc file giá vàng do người dùng
tự cung cấp, làm sạch dữ liệu và tính lợi suất. Repository pipeline không
lưu trữ hoặc phân phối dữ liệu giá vàng thô.

## Cách chạy

Các notebook đã được kiểm tra với **Python 3.14.5**. Mở bằng Jupyter Notebook
hoặc Google Colab và chạy các cell từ trên xuống. Chạy
`btc-sjc-vnindex-volatility.ipynb` trước để tái tạo
`outputs/model2_conditional_volatility.csv`, sau đó chạy
`tvp-var-connectedness.ipynb` nếu muốn thực hiện phần mở rộng TVP–VAR. Cell đầu
notebook cài đúng phiên bản các thư viện cần thiết; cũng có thể chuẩn bị môi
trường trước bằng `pip install -r requirements.txt`.

## Ý nghĩa kinh tế thực tiễn

- **Định lượng rủi ro riêng của từng tài sản:** Bitcoin có mức biến động trung bình cao nhất, trong khi vàng SJC có độ dai dẳng biến động rất lớn. VN-Index nhìn chung ổn định hơn nhưng có thể xuất hiện các đợt tăng vọt; tin xấu còn làm biến động của chỉ số tăng mạnh hơn tin tốt cùng độ lớn. Kết quả này hữu ích cho việc đặt hạn mức rủi ro, điều chỉnh tỷ trọng và xây dựng kịch bản stress test.
- **Hỗ trợ đa dạng hóa danh mục:** tương quan giữa BTC, vàng SJC và VN-Index nhìn chung dương nhưng rất thấp, cho thấy việc kết hợp ba tài sản có thể mang lại lợi ích đa dạng hóa trong giai đoạn nghiên cứu. Tuy nhiên, tương quan không hoàn toàn cố định và có thể tăng ở một số thời điểm, nên nhà đầu tư không nên xem lợi ích phòng hộ là bất biến.
- **Nhận diện kênh truyền rủi ro:** ở kỳ dự báo 10 ngày, mức liên kết toàn hệ thống tương đối thấp (`TCI = 4,17%`), nghĩa là biến động chủ yếu đến từ cú sốc riêng của từng thị trường. Dù vậy, VN-Index là bên truyền rủi ro ròng, còn vàng SJC là bên tiếp nhận mạnh nhất; kênh `VN-Index → SJC` trở nên rõ hơn khi kéo dài kỳ dự báo.
- **Phân biệt rủi ro ngắn hạn và dài hạn:** chỉ số kết nối tăng từ `1,45%` ở kỳ 5 ngày lên `7,63%` ở kỳ 20 ngày. Vì vậy, giám sát từng thị trường có thể phù hợp trong ngắn hạn, nhưng quản trị danh mục trung và dài hạn cần tính đến tác động lan truyền tích lũy giữa các tài sản.

Các kết luận trên phản ánh mẫu dữ liệu và đặc tả mô hình của dự án, có ý nghĩa hỗ trợ phân tích và quản trị rủi ro hơn là một khuyến nghị đầu tư trực tiếp.

## Hạn chế của nghiên cứu

- **Lịch giao dịch không đồng bộ:** Bitcoin được giao dịch liên tục 24/7, trong khi vàng SJC và VN-Index chỉ có dữ liệu vào những ngày giao dịch tương ứng. Việc đồng bộ ba chuỗi theo các ngày chung làm giảm số quan sát và có thể bỏ qua phản ứng của Bitcoin trong cuối tuần hoặc ngày nghỉ. Chênh lệch thời điểm ghi nhận giá cũng có thể ảnh hưởng đến kết quả ước lượng tương quan và lan truyền rủi ro.
- **Khoảng thời gian nghiên cứu còn hạn chế:** bộ dữ liệu chưa bao phủ các cú sốc hệ thống lớn như đại dịch COVID-19. Vì vậy, kết quả chủ yếu phản ánh hành vi của ba thị trường trong giai đoạn mẫu hiện tại và chưa cho thấy đầy đủ cách tương quan, biến động và cơ chế lan truyền thay đổi trong một cuộc khủng hoảng đặc biệt nghiêm trọng.
- **Độ mới của phương pháp chưa cao:** các mô hình ARMA–EGARCH, DCC/CCC–GARCH và Diebold–Yılmaz giải quyết phù hợp các câu hỏi về biến động, tương quan và lan truyền rủi ro, nhưng đều là những phương pháp đã được sử dụng rộng rãi trong nghiên cứu tài chính. Đóng góp của dự án vì thế chủ yếu nằm ở việc áp dụng và đối chiếu các mô hình trên ba thị trường trong bối cảnh Việt Nam, thay vì đề xuất một mô hình hoặc phương pháp ước lượng mới.
- **Giới hạn về thời gian và năng lực thực hiện:** do nguồn lực, thời gian nghiên cứu và kinh nghiệm của người thực hiện còn hạn chế, dự án chưa thể khảo sát toàn bộ đặc tả mô hình, phương pháp ước lượng và kiểm định độ vững có thể áp dụng. Một số lựa chọn trong quy trình phân tích vì vậy mang tính thực hành và có thể tiếp tục được cải thiện trong các nghiên cứu sau.

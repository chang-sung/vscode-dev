# Parts Data Parser 코드 구성

`Form1.cs`는 폼 초기화, 버튼, 트레이 아이콘과 목록 표시를 담당합니다. 파일 파싱과 DB 처리는 아래 클래스에서 담당합니다. 모두 기존 `log_parser` 네임스페이스를 사용합니다.

| 파일 | 담당 기능 |
| --- | --- |
| `Form1.cs` | 화면 이벤트, 백그라운드 실행 시작, 재고 결과 표시, 자동 종료 |
| `ParserSettings.cs` | 입력 파일 및 로그 경로, 서비스에 전달할 DB 연결 문자열 |
| `PartsImportRunner.cs` | 파일 존재 확인과 전체 처리 순서 |
| `StockService.cs` | 필수 파츠 조회, TSV 재고 집계, 재고 스냅샷 저장 및 현재 잔고 갱신 |
| `PriceImportService.cs` | EMS/MRO 가격 엑셀 판별, 파싱 및 DB 저장 |
| `EmsPurchaseImportService.cs` | EMS 구매 TSV 파싱, 필터 및 DB 저장 |
| `MroPurchaseImportService.cs` | MRO 구매 엑셀 파싱, 필터 및 DB 저장 |
| `PurchaseService.cs` | 구매 동기화, 상태 승급, 안전재고 자동구매 및 수동 Drop 로그 |
| `ImportFileReader.cs` | 공통 TSV 읽기, 따옴표 처리, 날짜 및 숫자 변환 |
| `ImportRows.cs` | 가격/EMS/MRO 행 데이터 모델 |
| `Logger.cs` | 일반 로그 및 구매 상태 로그 파일 저장 |
| `RequesterMailRegistry.cs` | 구매요청자 이메일 조회 |

## 실행 순서

폼을 로드하면 기존과 같이 백그라운드에서 오늘 요청된 수동 Drop 로그, 재고, 가격, EMS 구매, MRO 구매, EMS 동기화, MRO 동기화, 안전재고 자동구매 순서로 실행합니다. 동기화 후 상태 승급과 작업 완료 후 자동 종료 흐름도 유지합니다.

재고 서비스는 `EssentialStockLoaded`, `ParsedStockLoaded`, `MatchingStockLoaded` 이벤트로 결과 목록을 전달합니다. `Form1.UpdateStockList`가 `Invoke`로 UI 스레드에서 목록과 개수를 갱신하므로 서비스는 폼 컨트롤을 참조하지 않습니다.

수동 Drop 중복 키는 `Req_Date`, `Site`, `EQP_Div`, `Q_Code`, `Parts_Div`, `P_N`, `Req_Qty`의 조합을 그대로 사용하며, 기존 `PurchaseStatus_yyyy-MM-dd.txt`의 `DropKey` 기록과 호환됩니다.

## Visual Studio에 적용

1. `Form1.cs`를 교체하고 새로 추가된 `.cs` 파일 9개를 같은 WinForms 프로젝트에 포함합니다. `Logger.cs`와 `RequesterMailRegistry.cs`는 기존 파일을 계속 사용합니다.
2. 파일을 명시적으로 포함하는 프로젝트에서는 **프로젝트 → 추가 → 기존 항목**으로 새 파일을 추가합니다. SDK 형식 프로젝트에서 해당 폴더의 파일을 자동 포함한다면 별도 추가 작업이 필요하지 않습니다.
3. `Form1.Designer.cs`, `Form1.resx`, `Program.cs`와 기존 네임스페이스/컨트롤 이름은 그대로 사용합니다. DB를 사용하는 서비스 5개에는 `using MySql.Data.MySqlClient;`가 포함되어 있습니다. 기존 프로젝트의 `MySql.Data` NuGet 패키지와 Excel Interop 참조를 사용합니다. 프로젝트가 `MySqlConnector` 패키지를 사용한다면 해당 5개 파일의 지시문을 `using MySqlConnector;`로 맞춥니다.
4. 입력 파일/로그 경로 변경은 `ParserSettings.cs`에서, 기존 DB 연결 문자열 변경은 `Form1.cs`에서 합니다.
5. 프로젝트를 빌드한 뒤 기존 입력 파일로 재고 목록·개수, 가격/구매 DB 저장, 구매 동기화 및 자동 종료를 확인합니다. 같은 날 두 번 실행하여 Drop 로그 중복이 추가되지 않는지도 확인합니다.

## 검증 범위

분리한 C# 파일 10개의 문법 검사, 이동된 메서드 27개의 본문 비교, SQL 문자열 22개 비교, 재고 집계/DB 처리 본문 비교 및 실행 순서 검증을 통과했습니다. Drop 키 생성 본문과 DB 연결 설정도 원본과 동일합니다.

저장소에는 프로젝트 파일과 디자이너 파일이 없고 검증 환경에 .NET 컴파일러, 실제 DB 및 Excel이 없어 전체 프로젝트 빌드와 DB/Excel 연동 실행은 수행하지 않았습니다.

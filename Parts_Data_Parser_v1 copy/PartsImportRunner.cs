using System;
using System.IO;

namespace log_parser
{
    internal sealed class PartsImportRunner
    {
        private readonly ParserSettings settings;
        private readonly StockService stock;
        private readonly PriceImportService price;
        private readonly EmsPurchaseImportService ems;
        private readonly MroPurchaseImportService mro;
        private readonly PurchaseService purchase;

        public PartsImportRunner(ParserSettings settings, StockService stock)
        {
            this.settings = settings ?? throw new ArgumentNullException(nameof(settings));
            this.stock = stock ?? throw new ArgumentNullException(nameof(stock));
            price = new PriceImportService(settings.ConnectionString);
            ems = new EmsPurchaseImportService(settings.ConnectionString);
            mro = new MroPurchaseImportService(settings.ConnectionString);
            purchase = new PurchaseService(settings.ConnectionString);
        }

        public void Run()
        {
            purchase.LogTodayDropPurchases();

            ProcessIfExists("재고(EMS Stock)", settings.StockFilePath, stock.Execute);
            ProcessIfExists("가격(Price)", settings.PriceFilePath, price.Execute);
            ProcessIfExists("[EMS] 구매상태", settings.EmsPurchaseFilePath, ems.Execute);
            ProcessIfExists("[MRO] 구매상태", settings.MroPurchaseFilePath, mro.Execute);

            try
            {
                purchase.SyncEmsToPurchaseList();
                purchase.SyncMroToPurchaseList();
                Logger.Info("[안전재고 자동구매] 검사 시작");
                purchase.AutoCreatePurchaseRequestBySafetyStock();
                Logger.Info("[안전재고 자동구매] 검사 종료");
            }
            catch { /* 각 메서드에서 로그 출력됨 */ }
        }

        private void ProcessIfExists(string title, string path, Action<string> run)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                Logger.Info($"[{title}] 파일이 없습니다: {path}");
                return;
            }

            try
            {
                Logger.Info($"[{title}] 처리 시작: {path}");
                run(path); // 기존 파싱 함수 실행
                Logger.Info($"[{title}] 처리 완료");
            }
            catch (Exception ex)
            {
                Logger.Info($"[{title}] 처리 중 오류: {ex.Message}");
            }
        }
    }
}

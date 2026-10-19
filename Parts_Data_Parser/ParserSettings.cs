using System;

namespace log_parser
{
    internal sealed class ParserSettings
    {
        public string ConnectionString { get; }

        public ParserSettings(string connectionString)
        {
            ConnectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }

        public readonly string LogDirectory = @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\Logs";
        public readonly string StatusUpdateLogDirectory = @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\Logs\Logs_Purchase_Status_Update";
        public readonly string StockFilePath =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_stock_list\STOCK.csv";
        public readonly string PriceFilePath =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_price_list\price.xlsx";
        public readonly string EmsPurchaseFilePath =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_purchase_status_list\EMS.csv";
        public readonly string MroPurchaseFilePath =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_purchase_status_mro_list\MRO.xlsx";
    }
}

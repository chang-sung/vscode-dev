using System;

namespace log_parser
{
    internal sealed class PriceRow
    {
        public string Q_Code { get; set; } = "";  // 부품
        public string P_N { get; set; } = "";     // 업체P/N
        public string Site { get; set; } = "";    // 출고창고
        public decimal? Price { get; set; }       // 결산단가
        public string Parts_Div { get; set; } = "";// EMS=부품명, MRO=[제조사] 품명
    }

    internal sealed class PurchaseStatusRow
    {
        public string Req_Num { get; set; } = "";       // 의뢰번호
        public string ERP_PR { get; set; } = "";        // ERP PR
        public string Q_Code { get; set; } = "";        // 부품
        public string P_N { get; set; } = "";           // 업체P/N
        public string Order_Num { get; set; } = "";     // 발주번호
        public DateTime? Order_Date { get; set; }       // 발주일자
        public DateTime? Arrival_Date { get; set; }     // 입고일
        public string PO_Reject { get; set; } = "";     // PO Reject
        public decimal? Order_Qty { get; set; }    // 의뢰수량
        public string Requester { get; set; } = "";     // 구매요청자

        public int Order_Num_Seq { get; set; } = 0;   // 기본값 0
    }

    internal sealed class MroPurchaseStatusRow
    {
        public string MRO_Code { get; set; } = "";      // 엑셀: BQMS코드
        public string P_N { get; set; } = "";           // 엑셀: [제조사] 품명
        public string Order_Num { get; set; } = "";     // 엑셀: P/O No.
        public DateTime? Order_Date { get; set; }       // 엑셀: P/O일자
        public DateTime? Arrival_Date { get; set; }     // 엑셀: G/R일
        public decimal? Order_Qty { get; set; }         // 엑셀: P/O수량
        public string Buyer { get; set; } = "";         // 엑셀: 구매담당자

        public int Order_Num_Seq { get; set; } = 0;     // 그룹별 순번 (PK 일부)
    }
}

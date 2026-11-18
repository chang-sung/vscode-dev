using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Helper;

namespace MotionKit {
  public class ServoParam {
    //명사: 서보 파라미터, 파일이름
    //동사: 저장하기, 읽어오기
    public ServoParam(string _axisName) {
      filePath = Global.ParamPath + _axisName;
      fileHelper = new FileHelper(filePath);
      Load();
    }

    string filePath;
    Dictionary<ParamList, double> param = new Dictionary<ParamList, double>();
    FileHelper fileHelper;

    public double this[ParamList idx] {
      get { return param[idx]; }
      set { param[idx] = value; }
    }

    public void Load() {
      //파일 로드
      //string,string -> ParamList, double
      //파라미터 값이 존재하지 않으면??? 
      //  -> 해당 ParamList의 값은 100000 으로 기본값 설정

      Dictionary<string, string> dic = fileHelper.FileLoad();

      for (ParamList item = 0; item < ParamList.MAX; item++) {
        string key = item.ToString();
        double value = 0.0;

        if (dic.ContainsKey(key))
          double.TryParse(dic[key], out value);
        else
          value = 10000;

        if (param.ContainsKey(item) == false)
          param.Add(item, value);
      }
      Save();
    }
    public void Save() {
      //param을 file로 save
      fileHelper.FileSave(param);
    }
  }
}

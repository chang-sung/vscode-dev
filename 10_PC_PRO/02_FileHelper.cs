using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Helper {
  public class FileHelper {
    //포맷: item=value     ex) JogVel=10000
    //명사: path
    //동사: load, save

    public FileHelper(string _filePath) {
      filePath = _filePath + ".txt";
    }

    string filePath;

    public Dictionary<string, string> FileLoad() {
      Dictionary<string, string> dic = new Dictionary<string, string>();

      string path = Path.GetDirectoryName(filePath);
      if (Directory.Exists(path) == false)
        Directory.CreateDirectory(path);
      
      FileStream fs = new FileStream(filePath, FileMode.OpenOrCreate);
      StreamReader sr = new StreamReader(fs);

      while (sr.EndOfStream == false) {
        string key = "";
        string value = "";
        string[] strs = sr.ReadLine().Split('=');
        
        if(strs.Length == 2) {
          key = strs[0];
          value = strs[1];

          if (dic.ContainsKey(key) == false)
            dic.Add(key, value);
        }
      }

      sr.Close();
      fs.Close();

      return dic;
    }
    public void FileSave<T1,T2>(Dictionary<T1,T2> dic) {
      FileStream fs = new FileStream(filePath, FileMode.Create);
      StreamWriter sw = new StreamWriter(fs);
      StringBuilder sb = new StringBuilder();

      foreach (var item in dic) {
        //"JogVel=10000"
        string str = $"{item.Key}={item.Value}\r\n";
        sb.Append(str);
      }
      sw.Write(sb.ToString());
      sw.Close();
      fs.Close();
    }
  }
}

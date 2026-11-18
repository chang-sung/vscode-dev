using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MotionKit {
  public partial class frmPopup : Form {
    public frmPopup(string _axis, string _item, string _value) {
      InitializeComponent();
      this.Text = _axis;
      lbItem.Text = _item;
      tbValue.Text = _value;
    }

    double value;
    public double Value{ get { return value; } }
   
    private void btnOK_Click(object sender, EventArgs e) {
      if (double.TryParse(tbValue.Text, out value))
        this.DialogResult = DialogResult.OK; //DialogResult에 OK넣어주면 창이 닫힘
      else
        MessageBox.Show("잘못된 입력입니다.");
    }
  }
}

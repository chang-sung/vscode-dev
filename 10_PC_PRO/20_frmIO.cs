using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Helper;

namespace MotionKit {
  public partial class frmIO : Form {
    public frmIO() {
      InitializeComponent();
      CreateUI();
      io = IO.GetInstance();
    }

    CheckBox[,] cbsX = new CheckBox[4, 8];
    CheckBox[,] cbsY = new CheckBox[4, 8];

    IO io;


    void CreateUI() {
      int offset = 20;
      int width = 50, height = 20;
      int idx = 0;

      for (int i = 0; i < cbsX.GetLength(0); i++) {
        for (int j = 0; j < cbsX.GetLength(1); j++) {
          int xPos = offset + (offset + width) * j;
          int yPos = offset + (offset + height) * i;

          cbsX[i, j] = new CheckBox();
          cbsX[i, j].Location = new Point(xPos, yPos);
          cbsX[i, j].Size = new Size(width, height);
          cbsX[i, j].Text = "X" + idx.ToString();
          cbsX[i, j].Tag = idx;

          gbX.Controls.Add(cbsX[i, j]);

          cbsY[i, j] = new CheckBox();
          cbsY[i, j].Location = new Point(xPos, yPos);
          cbsY[i, j].Size = new Size(width, height);
          cbsY[i, j].Text = "Y" + idx.ToString();
          cbsY[i, j].Tag = idx;
          cbsY[i, j].Click += FrmIO_Click;

          gbY.Controls.Add(cbsY[i, j]);

          idx++;
        }
      }

      int gWidth = offset + (offset + width) * cbsX.GetLength(1);
      int gHeight = offset + (offset + height) * cbsX.GetLength(0);
      gbX.Size = new Size(gWidth, gHeight);
      gbY.Size = new Size(gWidth, gHeight);
      gbY.Location = new Point(gbX.Left, gbX.Bottom + offset);

      this.Size = new Size(gbY.Right + 50, gbY.Bottom + 50);
    }

    private void FrmIO_Click(object sender, EventArgs e) {
      CheckBox c = sender as CheckBox;
      if(c != null) {
        int bit = Convert.ToInt32(c.Tag);
        if (c.Checked)
          io.On(bit);
        else
          io.Off(bit);
      }
    }

    private void tmUIUpdate_Tick(object sender, EventArgs e) {
      for (int i = 0; i < Global.MAX_IO; i++) {
        int x = i % 8;
        int y = i / 8;

        cbsX[y, x].Checked = io.X[i];
        cbsY[y, x].Checked = io.Y[i];
      }
    }

    private void frmIO_Load(object sender, EventArgs e) {
      tmUIUpdate.Start();
    }
  }
}

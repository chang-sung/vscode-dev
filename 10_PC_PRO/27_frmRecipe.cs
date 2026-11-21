using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using Helper;


namespace MotionKit {
  using System.Collections.Generic;
  public partial class frmRecipe : Form {
    public frmRecipe() {
      InitializeComponent();
      livCreate.View = View.Details;
      LoadRecipes();
      

    }
    private readonly string recipeConfigPath = Path.GetFullPath(
    Path.Combine(Application.StartupPath,
                 Global.ParamPath,
                 Global.Recipe_Config_NAME + ".txt"));

    private readonly HashSet<string> deletedRecipeNames =
    new HashSet<string>(StringComparer.Ordinal);

    public event Action<string> RecipeSelected;



    private void btnCreate_Click(object sender, EventArgs e) {
      string recipeName = tbCreate.Text.Trim();

      if (recipeName.Length == 0)
      {
        MessageBox.Show("레시피명을 입력하세요.");
        tbCreate.Focus();
        return;
      }

      foreach (ListViewItem item in livCreate.Items)
      {
        if (string.Equals(item.SubItems[1].Text, recipeName,
                          StringComparison.OrdinalIgnoreCase))
        {
          MessageBox.Show("이미 등록된 레시피입니다.");
          return;
        }
      }

      AddRecipeRow(recipeName);
      tbCreate.Clear();
      tbCreate.Focus();
    }

    private void AddRecipeRow(string recipeName)
    {
      var item = new ListViewItem((livCreate.Items.Count + 1).ToString());
      item.SubItems.Add(recipeName);
      livCreate.Items.Add(item);
    }

    private void btnSave_Click(object sender, EventArgs e) {
      if (livCreate.Items.Count == 0)
      {
        MessageBox.Show("저장할 레시피가 없습니다.");
        return;
      }

      var recipeNames = new string[livCreate.Items.Count];

      for (int i = 0; i < livCreate.Items.Count; i++)
        recipeNames[i] = livCreate.Items[i].SubItems[1].Text;

      try
      {
        Directory.CreateDirectory(Path.GetDirectoryName(recipeConfigPath));

        var savedRecipes = new List<string>();

        if (File.Exists(recipeConfigPath))
          savedRecipes.AddRange(File.ReadAllLines(recipeConfigPath));

        var existingNames = new HashSet<string>(
            savedRecipes, StringComparer.OrdinalIgnoreCase);

        foreach (string recipeName in recipeNames)
        {
          if (existingNames.Add(recipeName))
            savedRecipes.Add(recipeName);
        }

        File.WriteAllLines(recipeConfigPath, savedRecipes);
        LoadRecipes();

        MessageBox.Show("레시피를 저장했습니다.");
      }
      catch (Exception ex)
      {
        MessageBox.Show("레시피 저장 실패\n" + ex.Message);
      }
    }

    private void LoadRecipes()
    {
      try
      {
        string[] recipeNames = File.Exists(recipeConfigPath)
            ? File.ReadAllLines(recipeConfigPath)
            : new string[0];

        livSelect.Items.Clear();
        cbxSelect.Items.Clear();

        foreach (string line in recipeNames)
        {
          string recipeName = line.Trim();
          if (recipeName.Length == 0)
            continue;

          var item = new ListViewItem((livSelect.Items.Count + 1).ToString());
          item.SubItems.Add(recipeName);
          item.Tag = recipeName;  // TXT에서 수정할 원래 이름
          livSelect.Items.Add(item);
          cbxSelect.Items.Add(recipeName);
        }
      }
      catch (Exception ex)
      {
        MessageBox.Show("레시피 불러오기 실패\n" + ex.Message);
      }
    }

    private void btnSelect_Click(object sender, EventArgs e)
    {
      string recipeName = cbxSelect.SelectedItem as string;

      if (string.IsNullOrEmpty(recipeName))
      {
        MessageBox.Show("적용할 레시피를 선택하세요.");
        return;
      }

      RecipeSelected?.Invoke(recipeName);
    }

    private void livSelect_DoubleClick(object sender, EventArgs e)
    {
      var hit = livSelect.HitTest(
          livSelect.PointToClient(Cursor.Position));

      if (hit.Item == null || hit.SubItem != hit.Item.SubItems[1])
        return;

      var cell = hit.SubItem;
      var editor = new TextBox
      {
        Text = cell.Text,
        Bounds = cell.Bounds,
        Font = livSelect.Font
      };

      bool cancelEdit = false;

      editor.Leave += (s, args) =>
      {
        string newName = editor.Text.Trim();

        if (!cancelEdit && newName.Length > 0)
          cell.Text = newName;

        livSelect.Controls.Remove(editor);
        editor.Dispose();
      };

      editor.KeyDown += (s, args) =>
      {
        if (args.KeyCode != Keys.Enter && args.KeyCode != Keys.Escape)
          return;

        args.SuppressKeyPress = true;
        cancelEdit = args.KeyCode == Keys.Escape;
        livSelect.Focus();
      };

      livSelect.Controls.Add(editor);
      editor.BringToFront();
      editor.Focus();
      editor.SelectAll();
    }

    private void btnModify_Click(object sender, EventArgs e) {

      try
      {
        var savedRecipes = new List<string>();

        if (File.Exists(recipeConfigPath))
          savedRecipes.AddRange(File.ReadAllLines(recipeConfigPath));

        var renamedRecipes = new Dictionary<string, string>(
            StringComparer.Ordinal);

        foreach (ListViewItem item in livSelect.Items)
        {
          string oldName = (string)item.Tag;
          string newName = item.SubItems[1].Text;

          if (!string.Equals(oldName, newName, StringComparison.Ordinal))
            renamedRecipes.Add(oldName, newName);
        }

        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < savedRecipes.Count; i++)
        {
          string name = savedRecipes[i].Trim();
          if (name.Length == 0)
            continue;

          string newName;
          if (renamedRecipes.TryGetValue(name, out newName))
          {
            name = newName;
            savedRecipes[i] = newName;
          }

          if (!names.Add(name))
          {
            MessageBox.Show("중복된 레시피명입니다: " + name);
            return;
          }
        }

        foreach (ListViewItem item in livCreate.Items)
        {
          string name = item.SubItems[1].Text;

          if (names.Add(name))
            savedRecipes.Add(name);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(recipeConfigPath));
        File.WriteAllLines(recipeConfigPath, savedRecipes);

        livCreate.Items.Clear();
        LoadRecipes();
        MessageBox.Show("레시피를 저장했습니다.");
      }
      catch (Exception ex)
      {
        MessageBox.Show("레시피 저장 실패\n" + ex.Message);
      }
    }

    private void btnDelete_Click(object sender, EventArgs e)
    {
      if (livSelect.SelectedItems.Count == 0)
      {
        MessageBox.Show("삭제할 레시피를 선택하세요.");
        return;
      }

      ListViewItem selectedItem = livSelect.SelectedItems[0];
      string savedName = (string)selectedItem.Tag;

      try
      {
        var recipes = new List<string>(File.ReadAllLines(recipeConfigPath));

        int index = recipes.FindIndex(line =>
            string.Equals(line.Trim(), savedName, StringComparison.Ordinal));

        if (index < 0)
        {
          MessageBox.Show("저장 파일에서 해당 레시피를 찾을 수 없습니다.");
          return;
        }

        recipes.RemoveAt(index);
        File.WriteAllLines(recipeConfigPath, recipes);

        livSelect.Items.Remove(selectedItem);
        cbxSelect.Items.Remove(savedName);

        for (int i = 0; i < livSelect.Items.Count; i++)
          livSelect.Items[i].Text = (i + 1).ToString();

        MessageBox.Show(
        this,
        "레시피가 삭제되었습니다.",
        "레시피 삭제",
        MessageBoxButtons.OK,
        MessageBoxIcon.Information);
      }
      catch (Exception ex)
      {
        MessageBox.Show("레시피 삭제 실패\n" + ex.Message);
      }
    }





  }
}

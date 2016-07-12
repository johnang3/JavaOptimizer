package angland.optimizer.var;

import java.util.ArrayList;
import java.util.List;

public class IndexedKey<VarKey> {

  private final VarKey varKey;
  private final int row;
  private final int col;
  private final int hashCode;


  public static <VarKey> IndexedKey<VarKey> scalarKey(VarKey varKey) {
    return new IndexedKey<>(varKey, -1, -1);
  }

  public static <VarKey> IndexedKey<VarKey> matrixKey(VarKey varKey, int row, int col) {

    return new IndexedKey<>(varKey, row, col);
  }

  public static <VarKey> List<IndexedKey<VarKey>> getAllMatrixKeys(VarKey varKey, int height,
      int width) {
    List<IndexedKey<VarKey>> varKeys = new ArrayList<>(height * width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        varKeys.add(matrixKey(varKey, i, j));
      }
    }
    return varKeys;
  }

  public IndexedKey(VarKey varKey, int row, int col) {
    super();
    this.varKey = varKey;
    this.row = row;
    this.col = col;
    final int prime = 631;
    int result = 1;
    result = prime * result + ((varKey == null) ? 0 : varKey.hashCode());
    result = prime * result + row;
    result = prime * result + col;
    this.hashCode = result;
  }

  public VarKey getVarKey() {
    return varKey;
  }

  public int getRow() {
    return row;
  }

  public int getCol() {
    return col;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    @SuppressWarnings("rawtypes")
    IndexedKey other = (IndexedKey) obj;
    if (hashCode != other.hashCode) return false;
    if (row != other.row) return false;
    if (varKey == null) {
      if (other.varKey != null) return false;
    } else if (!varKey.equals(other.varKey)) return false;
    return true;
  }

  public String toString() {
    return "(" + varKey + " " + row + " " + col + ")";
  }



}

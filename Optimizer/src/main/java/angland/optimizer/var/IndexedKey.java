package angland.optimizer.var;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class IndexedKey<VarKey> {

  private final VarKey varKey;
  private final List<Integer> indices;
  private final int hashCode;


  public static <VarKey> IndexedKey<VarKey> scalarKey(VarKey varKey) {
    return new IndexedKey<>(varKey, Collections.unmodifiableList(new ArrayList<>(0)));
  }

  public static <VarKey> IndexedKey<VarKey> matrixKey(VarKey varKey, int row, int col) {
    List<Integer> dim = new ArrayList<>();
    dim.add(row);
    dim.add(col);
    return new IndexedKey<>(varKey, Collections.unmodifiableList(dim));
  }

  public static <VarKey> List<IndexedKey<VarKey>> getAllMatrixKeys(VarKey varKey, int height,
      int width) {
    List<IndexedKey<VarKey>> varKeys = new ArrayList<>(height * width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        varKeys.add(matrixKey(varKey, height, width));
      }
    }
    return varKeys;
  }

  public IndexedKey(VarKey varKey, List<Integer> indices) {
    super();
    this.varKey = varKey;
    this.indices = indices;
    final int prime = 631;
    int result = 1;
    result = prime * result + ((varKey == null) ? 0 : varKey.hashCode());
    if (indices != null) {
      for (int i = 0; i < indices.size(); ++i) {
        result = prime * result + indices.get(i);
      }
    }
    this.hashCode = result;
  }

  public VarKey getVarKey() {
    return varKey;
  }

  public List<Integer> getIndices() {
    return indices;
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
    if (indices == null) {
      if (other.indices != null) return false;
    } else if (!indices.equals(other.indices)) return false;
    if (varKey == null) {
      if (other.varKey != null) return false;
    } else if (!varKey.equals(other.varKey)) return false;
    return true;
  }



}

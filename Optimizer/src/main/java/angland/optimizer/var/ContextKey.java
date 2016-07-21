package angland.optimizer.var;

public class ContextKey<VarKey> extends IndexedKey<VarKey> {

  private final int idx;

  public ContextKey(VarKey varKey, int row, int col, int idx) {
    super(varKey, row, col);
    this.idx = idx;
  }

  public int getIdx() {
    return idx;
  }

}

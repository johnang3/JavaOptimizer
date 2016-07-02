package angland.optimizer.var;



public class ArrayVectorValue<VarKey> extends ArrayMatrixValue<VarKey>
    implements
      IVectorValue<VarKey> {

  public ArrayVectorValue(int length, ScalarValue<VarKey>[] values) {
    super(length, 1, values);
  }

  public ScalarValue<VarKey> get(int idx) {
    return get(idx, 1);
  }

  public int getLength() {
    return getHeight();
  }


  public static class Builder<VarKey> extends ArrayMatrixValue.Builder<VarKey> {

    public Builder(int length) {
      super(length, 1);
    }

    public ScalarValue<VarKey> get(int idx) {
      return get(idx, 1);
    }

    public int getLength() {
      return getHeight();
    }

    public void set(int idx, ScalarValue<VarKey> calc) {
      set(idx, 1, calc);
    }

    public ArrayVectorValue<VarKey> build() {
      return new ArrayVectorValue<>(getLength(), values);
    }
  }


}

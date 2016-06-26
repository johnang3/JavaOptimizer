package angland.optimizer.var;

import java.util.Map;


public class ArrayVectorValue<VarKey> extends ArrayMatrixValue<VarKey> implements IVectorValue<VarKey> {

  public ArrayVectorValue(int length, ScalarValue<VarKey>[] values,
      Map<IndexedKey<VarKey>, Double> context) {
    super(length, 1, values, context);
  }

  public ScalarValue<VarKey> get(int idx) {
    return getCalculation(idx, 1);
  }

  public int getLength() {
    return getHeight();
  }


  public static class Builder<VarKey> extends ArrayMatrixValue.Builder<VarKey> {

    public Builder(int length) {
      super(length, 1);
    }

    public ScalarValue<VarKey> get(int idx) {
      return getCalculation(idx, 1);
    }

    public int getLength() {
      return getHeight();
    }

    public void set(int idx, ScalarValue<VarKey> calc) {
      set(idx, 1, calc);
    }

    public ArrayVectorValue<VarKey> build(Map<IndexedKey<VarKey>, Double> context) {
      return new ArrayVectorValue<>(getLength(), values, context);
    }
  }


}

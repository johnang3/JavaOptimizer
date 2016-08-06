package angland.optimizer.var;


public class KeyedDerivativeMapNode<VarKey> extends KeyedDerivative<VarKey> {

  public KeyedDerivativeMapNode<VarKey> next;


  public KeyedDerivativeMapNode(ContextKey<VarKey> key, double value) {
    super(key, value);
  }


  void setValue(double value) {
    this.value = value;
  }

  @Override
  public int hashCode() {
    return getKey().hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) return false;
    if (!(other instanceof KeyedDerivative)) {
      return false;
    }
    @SuppressWarnings("rawtypes")
    KeyedDerivative casted = (KeyedDerivative) other;
    return getKey().equals(casted.getKey());
  }

}

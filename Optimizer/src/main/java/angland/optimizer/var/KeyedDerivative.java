package angland.optimizer.var;

public class KeyedDerivative<VarKey> {

  private final IndexedKey<VarKey> key;
  private final double value;

  public KeyedDerivative(IndexedKey<VarKey> key, double value) {
    super();
    this.key = key;
    this.value = value;
  }

  public IndexedKey<VarKey> getKey() {
    return key;
  }

  public double getValue() {
    return value;
  }


}

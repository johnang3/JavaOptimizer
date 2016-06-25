package angland.optimizer.var;

import java.util.List;

public class VectorValue<VarType> {

  private final List<Calculation<VarType>> value;

  public VectorValue(List<Calculation<VarType>> value) {
    super();
    this.value = value;
  }

  public List<Calculation<VarType>> value() {
    return value;
  }

}

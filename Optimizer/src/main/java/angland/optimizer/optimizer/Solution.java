package angland.optimizer.optimizer;

import java.util.Map;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;

public class Solution<VarType> {

  private final Map<IndexedKey<VarType>, Double> context;
  private final ScalarValue<VarType> result;

  public Solution(Map<IndexedKey<VarType>, Double> context, ScalarValue<VarType> result) {
    super();
    this.context = context;
    this.result = result;
  }

  public Map<IndexedKey<VarType>, Double> getContext() {
    return context;
  }


  public ScalarValue<VarType> getResult() {
    return result;
  }



}

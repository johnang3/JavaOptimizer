package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;



public class Context<VarKey> {

  private final double[] values;
  private final ContextTemplate<VarKey> contextTemplate;

  public Context(ContextTemplate<VarKey> contextTemplate, double[] values) {
    super();
    this.contextTemplate = contextTemplate;
    this.values = values;
  }

  public double get(ContextKey<VarKey> key) {
    return values[key.getIdx()];
  }


  public ContextTemplate<VarKey> getContextTemplate() {
    return contextTemplate;
  }

  public Map<ContextKey<VarKey>, Double> asMap() {
    Map<ContextKey<VarKey>, Double> map = new HashMap<>();
    for (int i = 0; i < values.length; ++i) {
      map.put(contextTemplate.getContextKeys().get(i), values[i]);
    }
    return map;
  }



}

package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;

public interface VectorExpression<VarKey> {

  public ArrayVectorValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> cache);

  @SuppressWarnings("unchecked")
  public default ArrayVectorValue<VarKey> evaluateAndCache(Map<IndexedKey<VarKey>, Double> context,
      Map<Object, Object> partialSolutions) {
    return (ArrayVectorValue<VarKey>) partialSolutions.computeIfAbsent(this,
        k -> evaluate(context, partialSolutions));
  }

  public default ArrayVectorValue<VarKey> evaluate(Map<IndexedKey<VarKey>, Double> ctx) {
    return evaluateAndCache(ctx, new HashMap<>());
  }



}

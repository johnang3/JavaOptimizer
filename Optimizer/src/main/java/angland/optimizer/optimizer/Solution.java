package angland.optimizer.optimizer;

import java.util.Map;
import java.util.function.Function;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;

public class Solution<Result, VarType> {

  private final Map<IndexedKey<VarType>, Double> context;
  private final Result result;
  private final IScalarValue<VarType> objective;

  public Solution(Map<IndexedKey<VarType>, Double> context, Result result,
      IScalarValue<VarType> objective) {
    super();
    this.context = context;
    this.result = result;
    this.objective = objective;
  }

  public Solution(Map<IndexedKey<VarType>, Double> context,
      Function<Map<IndexedKey<VarType>, Double>, Result> getResult,
      Function<Result, IScalarValue<VarType>> getObjective) {
    super();
    this.context = context;
    this.result = getResult.apply(context);
    this.objective = getObjective.apply(result);
  }

  public Map<IndexedKey<VarType>, Double> getContext() {
    return context;
  }


  public Result getResult() {
    return result;
  }

  public IScalarValue<VarType> getObjective() {
    return objective;
  }



}

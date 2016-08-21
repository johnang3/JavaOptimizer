package angland.optimizer;

import java.util.Map;
import java.util.function.Function;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.scalar.Scalar;

public class Solution<Result, VarType> {

  private final Map<ContextKey<VarType>, Double> context;
  private final Result result;
  private final Scalar<VarType> objective;

  public Solution(Map<ContextKey<VarType>, Double> context, Result result,
      Scalar<VarType> objective) {
    super();
    this.context = context;
    this.result = result;
    this.objective = objective;
  }

  public Solution(Map<ContextKey<VarType>, Double> context,
      Function<Map<ContextKey<VarType>, Double>, Result> getResult,
      Function<Result, Scalar<VarType>> getObjective) {
    super();
    this.context = context;
    this.result = getResult.apply(context);
    this.objective = getObjective.apply(result);
  }

  public Map<ContextKey<VarType>, Double> getContext() {
    return context;
  }


  public Result getResult() {
    return result;
  }

  public Scalar<VarType> getObjective() {
    return objective;
  }



}

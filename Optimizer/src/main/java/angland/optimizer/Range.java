package angland.optimizer;

public class Range {

  private final double max;
  private final double min;

  public Range(double min, double max) {
    super();
    if (max <= min) {
      throw new IllegalArgumentException("Max must be greater than min.");
    }
    this.max = max;
    this.min = min;
  }

  public double getMax() {
    return max;
  }

  public double getMin() {
    return min;
  }



}

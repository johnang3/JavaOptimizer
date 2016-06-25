package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;

public class ContextTemplate<VarKey> {

  private Map<IndexedKey<VarKey>, Integer> keyToGradientIndex = new HashMap<>();

  public int getGradientIndex(IndexedKey<VarKey> key) {
    return keyToGradientIndex.get(key);
  }


}

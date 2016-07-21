package angland.optimizer.var;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ContextTemplate<VarKey> {

  private final List<ContextKey<VarKey>> idxToKey;
  private final Map<IndexedKey<VarKey>, ContextKey<VarKey>> supportedKeys;

  public ContextTemplate(List<IndexedKey<VarKey>> keys) {
    supportedKeys = new HashMap<>(keys.size(), 1);
    idxToKey = new ArrayList<>(keys.size());
    for (int i = 0; i < keys.size(); ++i) {
      IndexedKey<VarKey> key = keys.get(i);
      ContextKey<VarKey> contextKey =
          new ContextKey<>(key.getVarKey(), key.getRow(), key.getCol(), i);
      supportedKeys.put(key, contextKey);
      idxToKey.add(contextKey);
    }
  }


  public int size() {
    return idxToKey.size();
  }

  public ContextKey<VarKey> getKey(int i) {
    return idxToKey.get(i);
  }

  public static <VarKey> Context<VarKey> simpleContext(Map<IndexedKey<VarKey>, Double> varMapping) {
    return new ContextTemplate<>(varMapping.keySet().stream().collect(Collectors.toList()))
        .createContext(varMapping);
  }

  public Context<VarKey> randomContext() {
    double values[] = new double[supportedKeys.size()];
    for (int i = 0; i < values.length; ++i) {
      values[i] = 2 * Math.random() - 1;
    }
    return new Context<>(this, values);
  }

  public Context<VarKey> createContext(Map<? extends IndexedKey<VarKey>, Double> varMapping) {
    double values[] = new double[supportedKeys.size()];
    for (Map.Entry<? extends IndexedKey<VarKey>, Double> entry : varMapping.entrySet()) {
      IndexedKey<VarKey> k = entry.getKey();
      IndexedKey<VarKey> baseKey = new IndexedKey<>(k.getVarKey(), k.getRow(), k.getCol());
      values[supportedKeys.get(baseKey).getIdx()] = entry.getValue();
    }
    return new Context<>(this, values);
  }


  public ContextKey<VarKey> getContextKey(IndexedKey<VarKey> key) {
    return supportedKeys.get(key);
  }

  public List<ContextKey<VarKey>> getContextKeys() {
    return idxToKey;
  }

}

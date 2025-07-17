```java
public final void write(JSONSerializer serializer, Object object, Object fieldName, Type fieldType, int features)
        throws IOException {
    SerializeWriter out = serializer.out;

    Object[] array = (Object[]) object;

    if (object == null) {
        out.writeNull(SerializerFeature.WriteNullListAsEmpty);
        return;
    }

    int size = array.length;

    int end = size - 1;

    if (end == -1) {
        out.append("[]");
        return;
    }

    SerialContext context = serializer.context;
    serializer.setContext(context, object, fieldName, 0);

    try {
        Class<?> preClazz = null;
        ObjectSerializer preWriter = null;
        out.append('[');

        if (out.isEnabled(SerializerFeature.PrettyFormat)) {
            serializer.incrementIndent();
            serializer.println();
            for (int i = 0; i < size; ++i) {
                if (i != 0) {
                    out.write(',');
                    serializer.println();
                }
                serializer.write(array[i]);
            }
            serializer.decrementIdent();
            serializer.println();
            out.write(']');
            return;
        }

        for (int i = 0; i < end; ++i) {
            Object item = array[i];

            if (item == null) {
                out.append("null,");
            } else {
                if (serializer.containsReference(item)) {
                    serializer.writeReference(item);
                } else {
                    Class<?> clazz = item.getClass();

                    if (!isClassAllowed(clazz)) {
                        throw new SecurityException("Untrusted class detected: " + clazz.getName());
                    }

                    if (clazz == preClazz) {
                        preWriter.write(serializer, item, null, null, 0);
                    } else {
                        preClazz = clazz;
                        preWriter = serializer.getObjectWriter(clazz);
                        preWriter.write(serializer, item, null, null, 0);
                    }
                }
                out.append(',');
            }
        }

        Object item = array[end];

        if (item == null) {
            out.append("null]");
        } else {
            if (serializer.containsReference(item)) {
                serializer.writeReference(item);
            } else {
                Class<?> clazz = item.getClass();
                if (!isClassAllowed(clazz)) {
                    throw new SecurityException("Untrusted class detected: " + clazz.getName());
                }
                serializer.writeWithFieldName(item, end);
            }
            out.append(']');
        }
    } finally {
        serializer.context = context;
    }
}

private boolean isClassAllowed(Class<?> clazz) {
    // Allow core Java classes, primitives, arrays, and explicitly trusted packages.
    String className = clazz.getName();
    return className.startsWith("java.") || 
           className.startsWith("javax.") || 
           className.startsWith("com.example.trusted") || 
           clazz.isPrimitive() || 
           clazz.isArray();
}
```
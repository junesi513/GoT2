import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.parser.DefaultJSONParser;
import com.alibaba.fastjson.parser.JSONLexer;
import com.alibaba.fastjson.parser.JSONToken;
import com.alibaba.fastjson.parser.deserializer.ObjectDeserializer;
import com.alibaba.fastjson.util.TypeUtils;

import java.lang.reflect.*;
import java.util.List;

public class SafeDeserializer implements ObjectDeserializer {

    @SuppressWarnings("unchecked")
    public <T> T deserialze(DefaultJSONParser parser, Type type, Object fieldName) {
        final JSONLexer lexer = parser.lexer;
        if (lexer.token() == JSONToken.NULL) {
            lexer.nextToken(JSONToken.COMMA);
            return null;
        }

        if (lexer.token() == JSONToken.LITERAL_STRING) {
            byte[] bytes = lexer.bytesValue();
            lexer.nextToken(JSONToken.COMMA);
            return (T) bytes;
        }

        Type componentType;
        Class<?> componentClass;

        if (type instanceof GenericArrayType) {
            GenericArrayType genericArrayType = (GenericArrayType) type;
            componentType = genericArrayType.getGenericComponentType();
            componentClass = TypeUtils.getClass(componentType);

        } else if (type instanceof Class<?>) {
            Class<?> clazz = (Class<?>) type;
            componentType = clazz.getComponentType();
            if (componentType == null) {  // Not an array type
                // Handle cases where the input is not an array type
                JSONArray array = new JSONArray();
                parser.parseArray(array);
                return (T) toObjectArray(parser, componentClass, array);
            }
           componentClass = (Class<?>) componentType;
        } else {
            // Handle other unexpected types as needed
            throw new UnsupportedOperationException("Unsupported type: " + type.getTypeName());

        }

       if (componentClass == null) {
           componentClass = Object.class; // Fallback if type resolution fails
       }

        List<Object> list = new JSONArray();
        parser.parseArray(componentClass, list, fieldName); // Parse with resolved componentClass


        return (T) toObjectArray(parser, componentClass, list);
    }


    private <T> T toObjectArray(DefaultJSONParser parser, Class<?> componentClass, List<Object> list) {
        if (componentClass.isPrimitive()) {
            componentClass = TypeUtils.wrapperClass(componentClass);
        }
        Object[] array = (Object[]) Array.newInstance(componentClass, list.size());
        for (int i = 0; i < list.size(); i++) {
            Object item = list.get(i);
            if (item != null && item.getClass() != componentClass && !componentClass.isAssignableFrom(item.getClass())) {
              array[i] = TypeUtils.cast(item, componentClass, parser.getConfig());
            } else {
                array[i] = item;
            }
        }
        return (T) array;

    }

}
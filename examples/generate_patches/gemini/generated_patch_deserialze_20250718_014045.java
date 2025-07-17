import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.parser.DefaultJSONParser;
import com.alibaba.fastjson.parser.JSONLexer;
import com.alibaba.fastjson.parser.JSONToken;
import com.alibaba.fastjson.util.TypeUtils;

import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.List;

public class JSONDeserializer {

    public <T> T deserialze(DefaultJSONParser parser, Type type, Object fieldName) {
        final JSONLexer lexer = parser.lexer;
        if (lexer.token() == JSONToken.NULL) {
            lexer.nextToken(JSONToken.COMMA);
            return null;
        }

        if (lexer.token() == JSONToken.LITERAL_STRING) {
            byte[] bytes = lexer.bytesValue();
            lexer.nextToken(JSONToken.COMMA);
            if (type == byte[].class) {
                return (T) bytes;
            } else {
                // Handle potential mismatch, throw exception or return null based on your needs
                throw new ClassCastException("Type mismatch. Expected " + type + " but got byte[]");
            }
        }

        Class<?> componentClass;
        Type componentType;

        if (type instanceof GenericArrayType) {
            GenericArrayType genericArrayType = (GenericArrayType) type;
            componentType = genericArrayType.getGenericComponentType();
            componentClass = TypeUtils.getClass(componentType);
        } else if (type instanceof Class<?>) {
            Class<?> clazz = (Class<?>) type;
            componentType = clazz.getComponentType();
            if (componentType == null) {
                // Not an array type
                throw new ClassCastException("Type mismatch. Expected an array type but got " + type);
            }
            componentClass = componentType;
        } else {
            // Handle other types if needed, otherwise throw an exception
            throw new UnsupportedOperationException("Unsupported type: " + type);

        }


        List<?> list = parser.parseArray(componentClass, fieldName);

        if (list == null) {
            return null;
        }

        Object array = Array.newInstance(componentClass, list.size());

        for (int i = 0; i < list.size(); i++) {
            Object item = list.get(i);
            if (componentClass.isPrimitive()) {
                if (item instanceof Number) {
                    Array.set(array, i, TypeUtils.castToPrimitive(item));
                } else {
                    throw new ClassCastException("Cannot cast " + item.getClass() + " to " + componentClass);
                }
            } else if (!componentClass.isAssignableFrom(item.getClass()) && item != null){
                Array.set(array, i, TypeUtils.cast(item, componentClass, parser.getConfig()));
            } else {
                Array.set(array, i, item);
            }
        }

        return (T) array;
    }
}
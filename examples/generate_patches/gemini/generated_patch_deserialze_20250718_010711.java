import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONException;
import com.alibaba.fastjson.parser.DefaultJSONParser;
import com.alibaba.fastjson.parser.JSONLexer;
import com.alibaba.fastjson.parser.JSONToken;
import com.alibaba.fastjson.util.TypeUtils;

import java.lang.reflect.*;
import java.util.ArrayList;
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
                throw new JSONException("Type mismatch: expected byte array, got String");
            }

        }


        Class<?> componentClass = null;
        Type componentType;

        if (type instanceof GenericArrayType) {
            GenericArrayType gat = (GenericArrayType) type;
            componentType = gat.getGenericComponentType();
        } else if (type instanceof Class<?>) {
             Class<?> clazz = (Class<?>) type;
             if(clazz.isArray()){
                 componentType = clazz.getComponentType();
             } else {
                throw new JSONException("Type mismatch: expected array type");
             }
        } else{
            throw new JSONException("Unsupported type " + type.getTypeName());
        }
        
        if (componentType instanceof Class) {
            componentClass = (Class<?>) componentType;
        } else if (componentType instanceof ParameterizedType) {
             componentClass = (Class<?>) ((ParameterizedType) componentType).getRawType();
        } else if (componentType instanceof TypeVariable) {
            TypeVariable<?> tv = (TypeVariable<?>) componentType;
            Type[] bounds = tv.getBounds();
            if (bounds.length == 1 && bounds[0] instanceof Class) {
                 componentClass = (Class<?>) bounds[0];
            } else {
                componentClass = Object.class; // Default fallback
            }
        } else {
             componentClass = Object.class; // Default fallback
        }

        List<Object> list = new ArrayList<>();
        parser.parseArray(componentClass, list, fieldName);

        return (T) toObjectArray(parser, componentClass, list);

    }



    private Object toObjectArray(DefaultJSONParser parser, Class<?> componentType, List<Object> list) {
        if (componentType.isPrimitive()) {
            throw new JSONException("Primitive component type not supported for array conversion: " + componentType);
        }
        Object array = Array.newInstance(componentType, list.size());
        for (int i = 0; i < list.size(); i++) {
            Array.set(array, i, TypeUtils.cast(list.get(i), componentType, parser.getConfig()));
        }
        return array;
    }
}
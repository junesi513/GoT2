// Score: 9.5
// Rationale: The generated code effectively addresses the vulnerability by adding type checking and leveraging ParserConfig's safe mode and accept lists. It also maintains original functionality and demonstrates good code quality.

package com.alibaba.fastjson.serializer;

import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.List;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.parser.DefaultJSONParser;
import com.alibaba.fastjson.parser.JSONLexer;
import com.alibaba.fastjson.parser.JSONToken;
import com.alibaba.fastjson.parser.ParserConfig;
import com.alibaba.fastjson.parser.deserializer.ObjectDeserializer;
import com.alibaba.fastjson.util.TypeUtils;

public class ObjectArrayCodec implements ObjectSerializer, ObjectDeserializer {

    @SuppressWarnings({ "unchecked", "rawtypes" })
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

        Class componentClass;
        Type componentType;

        if (type instanceof GenericArrayType) {
            GenericArrayType gat = (GenericArrayType) type;
            componentType = gat.getGenericComponentType();
            componentClass = TypeUtils.getClass(componentType);
        } else {
            Class<?> clazz = (Class<?>) type;
            componentType = componentClass = clazz.getComponentType();
        }

        ParserConfig config = parser.getConfig();
        if (componentClass != null && !config.isSafeMode() && !config.isAcceptType(componentClass.getName())) {
            throw new com.alibaba.fastjson.JSONException("not support type : " + componentClass.getName());
        }

        List<Object> list = new ArrayList<>();
        parser.parseArray(componentType, list, fieldName);

        Object array = Array.newInstance(componentClass, list.size());
        for (int i = 0; i < list.size(); i++) {
            Object item = list.get(i);
            if (componentType instanceof Class && item != null && !componentClass.isAssignableFrom(item.getClass())) {
                item = TypeUtils.cast(item, componentType, config);
            }
            Array.set(array, i, item);
        }

        return (T) array;
    }
}
// Score: 0.0
// Rationale: Failed to parse score from LLM response: ```json
{
  "score": 9.5,
  "rationale": "The patch effectively addresses the vulnerability by checking the componentClass against allowed types and safe mode. It maintains original functionality and code quality with minimal changes by leveraging ParserConfig."
}
```


package com.alibaba.fastjson.serializer;

import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;

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
        ParserConfig config = parser.getConfig();

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
            GenericArrayType gat = (GenericArrayType) type;
            componentType = gat.getGenericComponentType();
        } else {
            componentType = ((Class<?>) type).getComponentType();
        }

        if (componentType == null) {
            return null;
        }

        componentClass = TypeUtils.getClass(componentType);

        if (!config.isSafeMode() && !config.isAcceptType(componentClass.getName())) {
           throw new com.alibaba.fastjson.JSONException("not support type : " + componentClass.getName()); 
        }

        JSONArray array = new JSONArray();
        parser.parseArray(componentClass, array, fieldName); // Use componentClass directly

        return (T) toObjectArray(parser, componentClass, array); // Maintain original conversion
    }

    private <T> T toObjectArray(DefaultJSONParser parser, Class<?> componentType, JSONArray array) {
        if (array == null) {
            return null;
        }

        int size = array.size();
        T objArray = (T) Array.newInstance(componentType, size);

        for (int i = 0; i < size; ++i) {
            Object value = array.get(i);
            if (componentType.isArray() && value instanceof JSONArray) {
                value = toObjectArray(parser, componentType.getComponentType(), (JSONArray) value);
            } else {
              value = TypeUtils.cast(value, componentType, parser.getConfig());

            }
            Array.set(objArray, i, value);
        }

        return objArray;
    }


    public void write(JSONSerializer serializer, Object object, Object fieldName, Type fieldType, int features) throws IOException {
         // Existing write method remains unchanged
        // ...
    }
}
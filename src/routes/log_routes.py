from fastapi import APIRouter, HTTPException, Query
from src.config.db import get_database
from src.utils.logger import logger
from typing import Optional, List
from datetime import datetime
from bson import ObjectId

router = APIRouter()

@router.get("", response_model=dict)
async def get_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    ai_model_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    success: Optional[bool] = None
):
    """
    Get paginated request logs with optional filtering.
    """
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
        
        query = {}
        
        # Filters
        if ai_model_id:
            try:
                query["ai_model_id"] = ObjectId(ai_model_id)
            except:
                pass # Ignore invalid object ids (or raise 400)

        if success is not None:
            query["success"] = success
            
        if start_date or end_date:
            date_filter = {}
            if start_date:
                try:
                    date_filter["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            if end_date:
                try:
                    date_filter["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            if date_filter:
                query["timestamp"] = date_filter
                
        # Pagination
        skip = (page - 1) * limit
        
        # Execute query
        total = await db.request_logs.count_documents(query)
        cursor = db.request_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        
        logs = []
        
        # Collect model IDs to fetch pricing
        log_list = []
        model_ids = set()
        
        async for log in cursor:
            # Convert ObjectIds to strings and datetimes to ISO format
            log["_id"] = str(log["_id"])
            if isinstance(log.get("ai_model_id"), ObjectId):
                log["ai_model_id"] = str(log["ai_model_id"])
            elif isinstance(log.get("ai_model_id"), str):
                 # It might already be a string, but just in case it's an ObjectId string representation
                 pass
            
            if log.get("ai_model_id"):
                model_ids.add(ObjectId(log["ai_model_id"]))
            
            if "timestamp" in log and isinstance(log["timestamp"], datetime):
                log["timestamp"] = log["timestamp"].isoformat()
            
            log_list.append(log)
            
        # Fetch models for pricing
        models = {}
        if model_ids:
            async for model in db.ai_models.find({"_id": {"$in": list(model_ids)}}):
                models[str(model["_id"])] = model
        
        # Calculate costs
        for log in log_list:
            model_id = log.get("ai_model_id")
            if model_id and model_id in models:
                model = models[model_id]
                cost_config = model.get("cost", {})
                
                input_tokens = log.get("input_tokens", 0)
                output_tokens = log.get("output_tokens", 0)
                
                input_cost = (input_tokens / 1_000_000) * cost_config.get("input_per_million", 0)
                output_cost = (output_tokens / 1_000_000) * cost_config.get("output_per_million", 0)
                
                log["cost"] = input_cost + output_cost
            else:
                log["cost"] = 0
            
            logs.append(log)
            
        return {
            "success": True,
            "data": logs,
            "meta": {
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (total + limit - 1) // limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats", response_model=dict)
async def get_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = Query("total_requests", regex="^(model_name|total_requests|total_tokens|total_cost|avg_response_time|successful_requests|failed_requests)$"),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$")
):
    """
    Get aggregated usage statistics (tokens, requests, latency) grouped by model.
    Supports sorting by: model_name, total_requests, total_tokens, total_cost, avg_response_time, successful_requests, failed_requests
    """
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=503, detail="Database not connected")
            
        pipeline = []
        
        # Date filter match stage
        match_stage = {}
        if start_date or end_date:
            date_query = {}
            if start_date:
                try:
                    date_query["$gte"] = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            if end_date:
                try:
                    date_query["$lte"] = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError:
                    pass
            if date_query:
                match_stage["timestamp"] = date_query
        
        if match_stage:
            pipeline.append({"$match": match_stage})
            
        # Group stage
        pipeline.append({
            "$group": {
                "_id": "$ai_model_id",
                "total_requests": {"$sum": 1},
                "total_input_tokens": {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "total_tokens": {"$sum": "$total_tokens"},
                "avg_response_time": {"$avg": "$response_time_ms"},
                "successful_requests": {
                    "$sum": {"$cond": [{"$eq": ["$success", True]}, 1, 0]}
                },
                "failed_requests": {
                    "$sum": {"$cond": [{"$eq": ["$success", False]}, 1, 0]}
                }
            }
        })
        
        # Lookup model details to get names
        pipeline.append({
            "$lookup": {
                "from": "ai_models",
                "localField": "_id",
                "foreignField": "_id",
                "as": "model_info"
            }
        })
        
        # Project stage to format output
        pipeline.append({
            "$project": {
                "_id": 0,
                "model_id": {"$toString": "$_id"},
                "model_name": {"$arrayElemAt": ["$model_info.name", 0]},
                "provider": {"$arrayElemAt": ["$model_info.provider", 0]}, # Optional
                "total_requests": 1,
                "total_input_tokens": 1,
                "total_output_tokens": 1,
                "total_tokens": 1,
                "avg_response_time": 1,
                "successful_requests": 1,
                "failed_requests": 1,
                "input_price": {"$ifNull": [{"$arrayElemAt": ["$model_info.cost.input_per_million", 0]}, 0]},
                "output_price": {"$ifNull": [{"$arrayElemAt": ["$model_info.cost.output_per_million", 0]}, 0]}
            }
        })
        
        # Add stage to calculate total_cost for sorting
        pipeline.append({
            "$addFields": {
                "total_cost": {
                    "$add": [
                        {"$divide": [{"$multiply": ["$total_input_tokens", "$input_price"]}, 1000000]},
                        {"$divide": [{"$multiply": ["$total_output_tokens", "$output_price"]}, 1000000]}
                    ]
                }
            }
        })
        
        # Sort stage
        sort_direction = -1 if sort_order == "desc" else 1
        pipeline.append({
            "$sort": {sort_by: sort_direction}
        })
        
        stats = []
        async for doc in db.request_logs.aggregate(pipeline):
            # Clean up temporary fields (total_cost is already calculated in pipeline)
            doc.pop("input_price", None)
            doc.pop("output_price", None)
            
            stats.append(doc)
            
        # Calculate global totals
        global_totals = {
            "total_requests": sum(s["total_requests"] for s in stats),
            "total_tokens": sum(s["total_tokens"] for s in stats),
            "input_tokens": sum(s["total_input_tokens"] for s in stats),
            "output_tokens": sum(s["total_output_tokens"] for s in stats),
            "total_cost": sum(s["total_cost"] for s in stats)
        }
            
        return {
            "success": True,
            "data": stats,
            "global_totals": global_totals
        }

    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

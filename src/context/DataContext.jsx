import React, { createContext, useContext, useState, useEffect } from 'react';
import { ticketService } from '../services/apiService';

const DataContext = createContext();

export function DataProvider({ children }) {
  const [tickets, setTickets] = useState([]);
  const [loading, setLoading] = useState(true);

  // Hardcode business rules as config instead of mock data
  const businessRules = [
    { id: 1, name: 'Auto-Resolve Low Risk Refunds', condition: 'Intent == Refund && Trust Score > 80', action: 'Auto-Refund', active: true, hits: 142 },
    { id: 2, name: 'Escalate Angry VIPs', condition: 'Sentiment == Angry && Tier == VIP', action: 'Escalate to Tier 2', active: true, hits: 28 },
    { id: 3, name: 'Flag Suspicious Keywords', condition: 'Entities includes ["sue", "lawyer"]', action: 'Hold for Review', active: true, hits: 12 },
    { id: 4, name: 'Draft Responses for Gen Inquiries', condition: 'Intent == Inquiry', action: 'Draft AI Response', active: true, hits: 512 }
  ];

  useEffect(() => {
    let mounted = true;
    const fetchTickets = async () => {
      try {
        const data = await ticketService.getTickets();
        if (mounted) setTickets(data);
      } catch (err) {
        console.error("Context fetch error:", err);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    
    fetchTickets();
    const interval = setInterval(fetchTickets, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <DataContext.Provider value={{ tickets, loading, businessRules }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  return useContext(DataContext);
}
